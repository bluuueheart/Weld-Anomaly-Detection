import torch
import torch.optim as optim
import torch.distributed as dist

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of the matrix G
    We assume G is a square matrix.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized Optimizer
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group.get('weight_decay', 0.0)

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                
                # Apply weight decay
                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                # State initialization
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize
                g_orth = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                # Scale by learning rate
                g_orth *= -lr

                if p.grad.ndim > 2:
                    g_orth = g_orth.view_as(p.grad)

                p.data.add_(g_orth.type_as(p.data))

        return loss

class MuonWithAuxAdam:
    """
    Wrapper class that uses Muon for parameters with use_muon=True and AdamW for others.
    """
    def __init__(self, param_groups):
        self.optimizers = []
        self.param_groups = param_groups
        
        muon_groups = []
        adam_groups = []
        
        for group in param_groups:
            # Extract use_muon flag
            use_muon = group.pop('use_muon', False)
            
            if use_muon:
                muon_groups.append(group)
            else:
                adam_groups.append(group)
        
        if muon_groups:
            self.optimizers.append(Muon(muon_groups))
        
        if adam_groups:
            self.optimizers.append(optim.AdamW(adam_groups))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for opt in self.optimizers:
            opt.step()
            
        return loss

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, state in zip(self.optimizers, state_dict):
            opt.load_state_dict(state)

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
