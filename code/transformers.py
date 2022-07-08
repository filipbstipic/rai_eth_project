from types import new_class
import torch

sigm = torch.sigmoid

class DeepPoly:
    def __init__(self, lower=None, upper=None, Al=None, bl=None, Au=None, bu=None):
        '''
        Box constraints: lower <= x <= upper
        Relational constraints: x <= Au x + bu and x >= Al x + bl
        '''
        dim = lower.shape.numel()
        
        self.lower = lower
        self.upper = upper

        self.Al = Al
        self.bl = bl
        if self.Al is None:
            self.Al = torch.zeros((dim, dim))
            self.bl = lower

        self.Au = Au
        self.bu = bu
        if self.Au is None:
            self.Au = torch.zeros((dim, dim))
            self.bu = upper

        self.history = []


class Normalization_Transformer(torch.nn.Module):

    def __init__(self, in_features):
        super(Normalization_Transformer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.mean = 0.1307
        self.sigma = 0.3081

    def forward(self, x):        
        x.lower = (x.lower - self.mean) / self.sigma
        x.upper = (x.upper - self.mean) / self.sigma
        x.bl = (x.bl - self.mean) / self.sigma
        x.bu = (x.bu - self.mean) / self.sigma
        return x


class Flatten_Transformer(torch.nn.Module):

    def __init__(self, in_features):
        super(Flatten_Transformer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x):
        dp_dict = {'lower': x.lower, 'upper': x.upper, 'Al': x.Al, 'bl': x.bl, 'Au': x.Au, 'bu': x.bu}
        x.history.append(dp_dict)

        x.lower = torch.flatten(x.lower).squeeze(0)
        x.upper = torch.flatten(x.upper).squeeze(0)
        x.bl = torch.flatten(x.bl).squeeze(0)
        x.bu = torch.flatten(x.bu).squeeze(0)

        return x


class Affine_Transformer(torch.nn.Module):

    def __init__(self, affine):

        assert isinstance(affine, torch.nn.modules.linear.Linear), 'Affine argument not an affine layer!'
        
        super(Affine_Transformer, self).__init__()
        self.A = affine.weight.detach()
        self.b = affine.bias.detach()
        self.in_features = affine.in_features
        self.out_features = affine.out_features

    def forward(self, x):
        dp_dict = {'lower': x.lower, 'upper': x.upper, 'Al': x.Al, 'bl': x.bl, 'Au': x.Au, 'bu': x.bu}
        x.history.append(dp_dict)

        x.Al = self.A
        x.Au = self.A
        x.bl = self.b
        x.bu = self.b
        x.lower, x.upper = self.backsubstitution(x)

        return x

    def backsubstitution(self, x):

        Au, Al, bu, bl = x.Au, x.Al, x.bu, x.bl

        for i in range(len(x.history) - 1, 1, -1):

            dp = x.history[i]

            zeros_matrix = torch.zeros_like(Al)

            Al_pos = torch.max(Al, zeros_matrix)
            Al_neg = torch.min(Al, zeros_matrix)
            
            Al = Al_pos @ dp['Al'] + Al_neg @ dp['Au']
            bl = Al_pos @ dp['bl'] + Al_neg @ dp['bu'] + bl

            Au_pos = torch.max(Au, zeros_matrix)
            Au_neg = torch.min(Au, zeros_matrix)

            Au = Au_pos @ dp['Au'] + Au_neg @ dp['Al']
            bu = Au_pos @ dp['bu'] + Au_neg @ dp['bl'] + bu

        # x.history contains at the beginning: [normalized output, flattened output, ...]
        # Here we get lower and upper bounds from flattened output, can vary this
        dp = x.history[1]

        zeros_matrix = torch.zeros_like(Al)
        
        lower = torch.max(Al, zeros_matrix) @ dp['lower'] + torch.min(Al, zeros_matrix) @ dp['upper'] + bl
        upper = torch.max(Au, zeros_matrix) @ dp['upper'] + torch.min(Au, zeros_matrix) @ dp['lower'] + bu

        return lower, upper


class SPU_transformer(torch.nn.Module):

    def __init__(self, in_features, random_restart=False):
        super(SPU_transformer, self).__init__()
        
        if random_restart:
            self.p = torch.nn.Parameter(data=4*(torch.rand(in_features) - 0.5), requires_grad=True)
        else:
            self.p = torch.nn.Parameter(data=torch.zeros(in_features), requires_grad=True)
        
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, x):
        dp_dict = {'lower': x.lower, 'upper': x.upper, 'Al': x.Al, 'bl': x.bl, 'Au': x.Au, 'bu': x.bu}
        x.history.append(dp_dict)

        lower, upper = x.lower, x.upper

        # We distinguish 3 cases: 
        # 1) 0 <= lower < upper    2) lower < upper <= 0      3) lower < 0 < upper

        case_1 = lower.ge(0)
        case_2 = upper.le(0)
        case_3 = ~(case_1 | case_2)

        # Initial q is slightly to the right of midpoint because of initialization
                
        q = torch.sigmoid(self.p  + 1e-4) * lower + (1 - torch.sigmoid(self.p  + 1e-4)) * upper
        
        # Compute lower, upper, Au, Al, bu, bl
        
        #############  CASE 1   #############
        
        Al_col =  case_1 * 2 * q
        x.bl =  case_1 * (- q**2 - 0.5)
        
        Au_col =  case_1 * (upper + lower)
        x.bu =  case_1 * (- upper * lower - 0.5)
        
        x.lower =  case_1 * (2 * lower * q - q**2 - 0.5)
        x.upper =  case_1 * (upper**2 - 0.5)
        
        #############  CASE 2   #############
        
        Al_col += case_2 * (sigm(-upper) - sigm(-lower))/(upper - lower + 1e-7)
        x.bl += case_2 * (-lower * (sigm(-upper) - sigm(-lower))/(upper - lower + 1e-7) + sigm(-lower) - 1)
        
        Au_col += case_2 * (sigm(-q) * (sigm(-q) - 1))
        x.bu += case_2 * (-q * sigm(-q) * (sigm(-q) - 1) + sigm(-q) - 1)
        
        x.lower += case_2 * (sigm(-upper) - 1)
        x.upper += case_2 * ( (lower - q) * sigm(-q) * (sigm(-q) - 1) + sigm(-q) - 1 )
        
        #############  CASE 3   #############
        
        # Lower inequality is special for case 3: we interpolate between 
        # 2*q*x - q**2 - 0.5 (for q in [0, u]) and x*(sigm(-q) - 0.5)/l - 0.5 (for q in [l, 0])
        # q is initialized to 0, we make it slightly bigger

        Au_segment = (upper**2 + 0.5 - sigm(-lower))/(upper - lower +  1e-7)
        Au_tangent = sigm(-lower) * (sigm(-lower) - 1)
        
        bu_segment = - lower * (upper**2 - sigm(-lower) + 0.5)/(upper - lower +  1e-7) + (sigm(-lower) - 1)
        bu_tangent = - lower * sigm(-lower)*(sigm(-lower) - 1) + (sigm(-lower) - 1)
        
        Au_col += case_3 * torch.where(Au_segment >= Au_tangent, Au_segment, Au_tangent)
        x.bu +=  case_3 * torch.where(Au_segment >= Au_tangent, bu_segment, bu_tangent)
        
        Al_col += case_3 * (2 * torch.max(q, torch.zeros_like(q)) +  \
                            (sigm(-torch.min(q, torch.zeros_like(q))) - 0.5) / (lower +  1e-7))
        x.bl += case_3 * (-torch.max(q, torch.zeros_like(q))**2 - 0.5)

        x.lower += case_3 * ( torch.max(q, torch.zeros_like(q)) * 2 * lower - torch.max(q, torch.zeros_like(q))**2 - 0.5 + \
                                upper * (sigm(- torch.min(q, torch.zeros_like(q)))-0.5) / (lower + 1e-7))
        x.upper += case_3 * torch.max(sigm(-lower) - 1, upper**2 - 0.5)
        
        # Transform into matrix
        
        x.Au = torch.diag(Au_col)
        x.Al = torch.diag(Al_col)
        
        return x


class Final_Verfication_Transformer(torch.nn.Module):

    def __init__(self, in_features, true_label):
        super(Final_Verfication_Transformer, self).__init__()
        self.true_label = true_label
        self.in_features = in_features
        self.out_features = in_features - 1

        # After last layer add additional affine layer
        # to check: x_true_label - x_i > 0
        A = torch.zeros((self.in_features, self.in_features))
        A[:, self.true_label] = 1
        A = -1 * torch.eye(self.in_features) + A
        
        self.A = torch.cat((A[:self.true_label], A[self.true_label+1:]))
        self.b = torch.zeros(self.in_features - 1)

    def forward(self, x):
        dp_dict = {'lower': x.lower, 'upper': x.upper, 'Al': x.Al, 'bl': x.bl, 'Au': x.Au, 'bu': x.bu}
        x.history.append(dp_dict)

        x.Al, x.Au = self.A, self.A
        x.bl, x.bu = self.b, self.b
        x.lower, x.upper = self.backsubstitution(x)

        return x.lower, x.upper

    def backsubstitution(self, x):

        Au, Al, bu, bl = x.Au, x.Al, x.bu, x.bl

        zeros_matrix = torch.zeros_like(Al)

        for i in range(len(x.history)-1, 1, -1):
            dp = x.history[i]

            zeros_matrix = torch.zeros_like(Al)

            Al_pos = torch.max(Al, zeros_matrix)
            Al_neg = torch.min(Al, zeros_matrix)
            
            Al = Al_pos @ dp['Al'] + Al_neg @ dp['Au']
            bl = Al_pos @ dp['bl'] + Al_neg @ dp['bu'] + bl

            Au_pos = torch.max(Au, zeros_matrix)
            Au_neg = torch.min(Au, zeros_matrix)

            Au = Au_pos @ dp['Au'] + Au_neg @ dp['Al']
            bu = Au_pos @ dp['bu'] + Au_neg @ dp['bl'] + bu

        # x.history contains at the beginning: [normalized output, flattened output, ...]
        # Here we get lower and upper bounds from flattened output, can vary this
        dp = x.history[1]

        zeros_matrix = torch.zeros_like(Al)
        lower = torch.max(Al, zeros_matrix) @ dp['lower'] + torch.min(Al, zeros_matrix) @ dp['upper'] + bl
        upper = torch.max(Au, zeros_matrix) @ dp['upper'] + torch.min(Au, zeros_matrix) @ dp['lower'] + bu

        return lower, upper

    