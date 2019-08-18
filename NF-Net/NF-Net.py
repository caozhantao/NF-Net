import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np 
import os
import argparse
from data.medical import MEDICAL
import torchvision
from utils.utilsinfo import *
from models.alexnet import alexnet
from torch.nn import functional as F

parser=argparse.ArgumentParser()
parser.add_argument('--num_workers',type=int,default=2)
parser.add_argument('--batchSize',type=int,default=16)
parser.add_argument('--nepoch',type=int,default=20)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
print(opt)
#os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]

transform_train=transforms.Compose([
	transforms.RandomCrop(224),  
	transforms.ToTensor(),
])

transform_val=transforms.Compose([ 
	torchvision.transforms.Resize(227),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
])

trainset = MEDICAL(root='./data/alldata/',
                                download=False,  
                                train=True, 
                                transform=transform_train,
                                noise_type='test',
				noise_rate=0.0
    			    )

valset = MEDICAL(root='./data/alldata/',
							download=False,  
							train=False, 
							transform=transform_val,
							noise_type='test',
			noise_rate=0.0
			)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)


model=alexnet(pretrained=True)
model.cuda()


ignored_params = list(map(id, model.classifier.parameters())) #layer need to be trained
base_params = filter(lambda p: id(p) not in ignored_params,model.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.0001},
    {'params': model.classifier.parameters()}], 0.0001, momentum=0.9, weight_decay=1e-3)

	
model2=alexnet(pretrained=True)

model2.cuda()




ignored_params = list(map(id, model2.classifier.parameters())) #layer need to be trained
base_params2 = filter(lambda p: id(p) not in ignored_params,model2.parameters())
optimizer2 = optim.SGD([
    {'params': base_params2, 'lr': 0.0001},
    {'params': model2.classifier.parameters()}], 0.01, momentum=0.9, weight_decay=1e-3)

scheduler=StepLR(optimizer,step_size=10, gamma=0.001)
criterion=nn.CrossEntropyLoss()
criterion.cuda()

scheduler2=StepLR(optimizer2,step_size=3, gamma=0.1)
criterion2=nn.CrossEntropyLoss()
criterion2.cuda()

def log_cls_bal(pred, tgt):
    return -torch.log(pred + 1.0e-6)

def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

def compute_aug_loss(stu_out, tea_out, clean, label):
		# Augmentation loss
		confidence_thresh = 0.96837722
		#confidence_thresh = 0.97
		n_classes = 2
		use_rampup = False
		rampup = 0
		
		
		conf_tea = torch.max(tea_out, 1)[0]
		#conf_tea = torch.max(stu_out, 1)[0]
		
		unsup_mask = conf_mask = torch.gt(conf_tea, confidence_thresh).float()
	
		unsup_mask_count = conf_mask_count = torch.sum(conf_mask)

	
		d_aug_loss = stu_out - tea_out
		aug_loss = d_aug_loss * d_aug_loss
	

		aug_loss = torch.mean(aug_loss, 1)
		
	
		unsup_loss = torch.mean(aug_loss * unsup_mask) * 3.0
	

		# Class balance loss
		cls_balance = 0.005

		if cls_balance > 0.0:
	
			#avg_cls_prob = torch.mean(stu_out, 0)
			avg_cls_prob = torch.mean(tea_out, 0)

			#equalise_cls_loss = robust_binary_crossentropy(avg_cls_prob, float(1.0 / n_classes))
			equalise_cls_loss = log_cls_bal(avg_cls_prob, float(1.0 / n_classes))


			equalise_cls_loss = torch.mean(equalise_cls_loss) * n_classes


			if use_rampup:
				equalise_cls_loss = equalise_cls_loss * rampup_weight_in_list[0]
			else:
				if rampup == 0:
					equalise_cls_loss = equalise_cls_loss * torch.mean(unsup_mask, 0)


			unsup_loss += equalise_cls_loss * cls_balance * -3.0

		return unsup_loss, conf_mask_count, unsup_mask_count

def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	model.train()

	scheduler2.step()
	model2.train()
	for batch_idx,(img,label,index, clean) in enumerate(trainloader):
    		ind=index.cpu().numpy().transpose()
		label = label.squeeze()

		image=Variable(img.cuda())
		label=Variable(label.cuda())

		optimizer.zero_grad()
		optimizer2.zero_grad()

		teacher_logits_out=model2(image)
		teacher_prob_out = F.softmax(teacher_logits_out ) 

		student_logits_out=model(image)
		student_prob_out = F.softmax(student_logits_out )

		teacher_loss=criterion2(teacher_logits_out ,label)

		student_loss=criterion(student_prob_out,label)
		
		unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_prob_out, teacher_prob_out, clean, label)

		loss_expr = student_loss + teacher_loss + unsup_loss


		loss_expr.backward()
		
		optimizer2.step()
		optimizer.step()



def val(epoch):
    	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	model2.eval()
	#for i in model2.named_parameters():
    		#pass
		#print (i)
	total=0
	total2=0
	correct2=0
	correct=0
	total_malignant1 = 0
	total_benign1 = 0
	pre_malignant1 = 0
	pre_benign1 = 0
	correct_malignant1 = 0
	correct_benign1 = 0

	total_malignant2 = 0
	total_benign2 = 0
	pre_malignant2 = 0
	pre_benign2 = 0
	correct_malignant2 = 0
	correct_benign2 = 0

	statistics_dict={}
	statistics_dict2={}

	for i in range(0, statistic_type.statistic_type_max):
		statistics_dict[i] = 0
		statistics_dict2[i] = 0

	with torch.no_grad():
		for batch_idx,(img,label,index, clean) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()

			statistics_result(predicted, label, statistics_dict)
    						
			image2=Variable(img.cuda())
			label2=Variable(label.cuda())
			out2=model2(image2)
			_,predicted2=torch.max(out2.data,1)
			total2+=image2.size(0)
			correct2+=predicted2.data.eq(label2.data).cpu().sum()

			statistics_result(predicted2, label, statistics_dict2)

	print(" Student Acc1: %f , Teacher Acc2: %f"% ((1.0*correct.numpy())/total, (1.0*correct2.numpy())/total2))

	compute_result("Student", statistics_dict)
	compute_result( "teacher", statistics_dict2)
	


			 
for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)

torch.save(model.state_dict(),'models/alexnet/my_alexnet_alldata_model.pth')
torch.save(model2.state_dict(),'models/alexnet/my_alexnet_alldata_model2.pth')
