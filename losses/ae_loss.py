import torch
import torch.nn as nn
import numpy as np

def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp

class AELoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_objs in joints:
            tmp = []
            for joint in joints_per_objs:
                # if joint[1] > 0:
                    # tmp.append(pred_tag[joint[0]])
                tmp.append(pred_tag[0][joint[0],joint[1]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return make_input(torch.zeros(1).float()), \
                make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), \
                pull/(num_tags)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, pull/(num_tags)

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
          #  print(tags[i].size(), joints[i].shape)
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)

# loss_f = AELoss('max')

# AE_pd  = torch.ones(2,1,256,256).cuda()
# Joints = [[[[100,100],[120,120]],[[110,110],[130,130]]],[[[100,100],[120,120]],[[110,110],[130,130]]]]
# Joints = np.array(Joints)
# Joints = torch.tensor(Joints).cuda()

# batch_size = AE_pd.size()[0]
# AE_pd      = AE_pd.contiguous().view(batch_size, -1, 1)

# loss_push, loss_pull = loss_f(AE_pd, Joints)
# print(Joints.shape)
# print(loss_push, loss_pull)



