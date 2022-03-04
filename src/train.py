from sklearn import metrics
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import time
softmax = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(path,min_test_epoch_loss,device_train,device_test,
                    epoch_num,train_loader,test_loader,
                    write_log,cls,optimizer,
                    test_epoch_loss_l,test_acc_l,train_epoch_loss_l,train_acc_l):
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch"+str(epoch_num)+" start_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    cls.to(device_train)
    cls.train()

    epoch_loss=0
    total=0
    correct=0
    output_all=[]
    label_all=[]
    for batch_idx,batch in enumerate(train_loader):
        label=batch['label'].to(device_train)#batch size * 1
        label_all.append(label.view(-1,1))
        input_ids=torch.stack(batch['input_ids']).t().to(device_train)#batch size * 100
        attention_mask=torch.stack(batch['attention_mask']).t().to(device_train)#batch size * 100

        #计算输出
        output = cls(input_ids, attention_mask=attention_mask)#batch size * 1

        #计算loss
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            #四舍五入
            output=softmax(output)
            output=output.argmax(dim=1)
            output_all.append(output)
            total+=len(output)
            
            #epoch_loss
            epoch_loss+=loss
            ave_loss=epoch_loss/total
            
            #计算准确率
            add_correct=(output== label).sum().item()
            correct+=add_correct
            acc=correct/total
            
            if batch_idx%5==0:
                print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    batch_idx, len(train_loader),100.*batch_idx/len(train_loader), 
                    correct, total,acc,
                    ave_loss
                    ),end= "\r")
            
            
            
    #结束：
    
#     can't convert cuda:5 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    with torch.no_grad():
        output_all=torch.cat(output_all,0)
        label_all=torch.cat(label_all,0)

        output_all=np.array(output_all.cpu())
        label_all=np.array(label_all.cpu())
        acc_score=metrics.accuracy_score(label_all,output_all)
        
#     print(metrics.classification_report(label_all,output_all))
    write_log('__________________train end__________________')
    write_log('正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    correct, total,acc,
                    ave_loss))
    
    write_log("准确率:"+str(acc_score))
    
    write_log('__________________test start__________________')
    test_acc,test_epoch_loss=test(device_test,test_loader,write_log,cls)
    if min_test_epoch_loss>test_epoch_loss:
        min_test_epoch_loss=test_epoch_loss
        write_log("store model")
        end = time.time()
        # torch.save(cls,path+"cls_"+str(epoch_num)+"_"+str(round(test_acc,5))+'_'+str(round(test_epoch_loss,5))+'_'+str(end)+".model")
    
    write_log('train_acc:'+str(acc)+'  train_epoch_loss:'+str(epoch_loss.item())+'  test_acc:'+str(test_acc)+'  test_epoch_loss:'+str(test_epoch_loss))
    write_log(str(acc)+'\t'+str(epoch_loss.item())+'\t'+str(test_acc)+'\t'+str(test_epoch_loss))
    write_log('__________________test end__________________')
    
    train_acc_l.append(acc)
    train_epoch_loss_l.append(epoch_loss.item())
    test_acc_l.append(test_acc)
    test_epoch_loss_l.append(test_epoch_loss)
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch "+str(epoch_num)+" end_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    return min_test_epoch_loss




def badloss(correct_bi,output_sm):
#     print('badloss')
    loss=0
    for i in range(len(output_sm)):
#         print(correct_bi[i].item())
        if correct_bi[i].item() is False:
            max_index=output_sm[i].argmax(dim=0)
            # print(max_index,output_sm[i][max_index])
            loss=loss-torch.log(1-output_sm[i][max_index])
#             print(0)
        else:
#             print(1)
            continue
#     print('bad_loss:',loss)
    return loss


def train_one_epoch_loss(path,min_test_epoch_loss,device_train,device_test,
                    epoch_num,train_loader,test_loader,
                    write_log,cls,optimizer,
                    test_epoch_loss_l,test_acc_l,train_epoch_loss_l,train_acc_l,lmd):
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch"+str(epoch_num)+" start_______________")
    write_log("_______________lmd="+str(lmd)+"_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    cls.to(device_train)
    cls.train()
    loss_mode=0
    epoch_loss=0
    total=0
    correct=0
    batch_correct=0
    acc=0
    output_all=[]
    label_all=[]
    for batch_idx,batch in enumerate(train_loader):
        label=batch['label'].to(device_train)#batch size * 1
        label_all.append(label.view(-1,1))
        input_ids=torch.stack(batch['input_ids']).t().to(device_train)#batch size * 100
        attention_mask=torch.stack(batch['attention_mask']).t().to(device_train)#batch size * 100

        #计算输出
        output = cls(input_ids, attention_mask=attention_mask)#batch size * 1
        output_sm=softmax(output)
        
        with torch.no_grad():
            output_end=output_sm.argmax(dim=1)
            correct_bi= output_end == label
#             print(correct_bi)
            add_correct=(correct_bi).sum().item()
            batch_correct=add_correct/len(output)
#             print(batch_correct)
        
        #计算loss
        loss_c = criterion(output, label)
        if batch_correct<0.9:
            loss = loss_c
        else:
            loss_mode+=1
            loss = loss_c + badloss(correct_bi,output_sm)/lmd
#         print('loss:',loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            #四舍五入
            output_all.append(output_end)
            
            #epoch_loss
            total+=len(output)
            epoch_loss += loss_c
            ave_loss=epoch_loss/total
            
            #计算准确率
            correct+=add_correct
            acc=correct/total
            
            if batch_idx%5==0:
                print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}, mode: {}____'.format(
                    batch_idx, len(train_loader),100.*batch_idx/len(train_loader),
                    correct, total, acc,
                    ave_loss, loss_mode
                    ),end= "\r")
                loss_mode=0
            
            
            
    #结束：
    
#     can't convert cuda:5 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    with torch.no_grad():
        output_all=torch.cat(output_all,0)
        label_all=torch.cat(label_all,0)

        output_all=np.array(output_all.cpu())
        label_all=np.array(label_all.cpu())
        acc_score=metrics.accuracy_score(label_all,output_all)
        
#     print(metrics.classification_report(label_all,output_all))
    write_log('__________________train end__________________')
    write_log('正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    correct, total,acc,
                    ave_loss))
    
    write_log("准确率:"+str(acc_score))
    
    write_log('__________________test start__________________')
    test_acc,test_epoch_loss=test(device_test,test_loader,write_log,cls)
    if min_test_epoch_loss>test_epoch_loss:
        min_test_epoch_loss=test_epoch_loss
        write_log("store model")
        # end = time.time()
        # torch.save(cls,path+"cls_loss_"+str(epoch_num)+"_"+str(round(test_acc,5))+'_'+str(round(test_epoch_loss,5))+'_'+str(end)+".model")
    
    write_log('train_acc:'+str(acc)+'  train_epoch_loss:'+str(epoch_loss.item())+'  test_acc:'+str(test_acc)+'  test_epoch_loss:'+str(test_epoch_loss))
    write_log(str(round(acc,4))+'\t'+str(round(epoch_loss.item(),4))+'\t'+str(round(test_acc,4))+'\t'+str(round(test_epoch_loss,4)))
    write_log('__________________test end__________________')
    
    train_acc_l.append(acc)
    train_epoch_loss_l.append(epoch_loss.item())
    test_acc_l.append(test_acc)
    test_epoch_loss_l.append(test_epoch_loss)
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch "+str(epoch_num)+" end_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    return min_test_epoch_loss
    

import math
def train_one_epoch_ct(path,min_test_epoch_loss,device_train,device_test,
                    epoch_num,train_loader,test_loader,
                    write_log,cls,optimizer,
                    test_epoch_loss_l,test_acc_l,train_epoch_loss_l,train_acc_l,criterion_ct,lmd):
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch"+str(epoch_num)+" start_______________")
    write_log("_______________lmd="+str(lmd)+"_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    cls.to(device_train)
    cls.train()

    epoch_loss=0
    total=0
    correct=0
    output_all=[]
    label_all=[]
    for batch_idx,batch in enumerate(train_loader):
        label=batch['label'].to(device_train)#batch size * 1
        label_all.append(label.view(-1,1))
        input_ids=torch.stack(batch['input_ids']).t().to(device_train)#batch size * 100
        attention_mask=torch.stack(batch['attention_mask']).t().to(device_train)#batch size * 100

        #计算输出
        output = cls(input_ids, attention_mask=attention_mask)#batch size * 1

        #计算loss
        loss = criterion(output, label)
        loss_ct = criterion_ct(device_train,output, label)
        if math.isinf(loss_ct) or math.isnan(loss_ct):
            loss_all=loss
        else:
            loss_all = loss+loss_ct/lmd
        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
            
        with torch.no_grad():
            #四舍五入
            output=softmax(output)
            output=output.argmax(dim=1)
            output_all.append(output)
            total+=len(output)
            
            #epoch_loss
            epoch_loss+=loss
            ave_loss=epoch_loss/total
            
            #计算准确率
            add_correct=(output== label).sum().item()
            correct+=add_correct
            acc=correct/total
            
            if batch_idx%5==0:
                print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    batch_idx, len(train_loader),100.*batch_idx/len(train_loader), 
                    correct, total,acc,
                    ave_loss
                    ),end= "\r")
            
            
            
    #结束：
    
#     can't convert cuda:5 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    with torch.no_grad():
        output_all=torch.cat(output_all,0)
        label_all=torch.cat(label_all,0)

        output_all=np.array(output_all.cpu())
        label_all=np.array(label_all.cpu())
        acc_score=metrics.accuracy_score(label_all,output_all)
        
#     print(metrics.classification_report(label_all,output_all))
    write_log('__________________train end__________________')
    write_log('正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    correct, total,acc,
                    ave_loss))
    
    write_log("准确率:"+str(acc_score))
    
    write_log('__________________test start__________________')
    test_acc,test_epoch_loss=test(device_test,test_loader,write_log,cls)
    if min_test_epoch_loss>test_epoch_loss:
        min_test_epoch_loss=test_epoch_loss
        write_log("store model")
        end = time.time()
        # torch.save(cls,path+"cls_"+str(epoch_num)+"_"+str(round(test_acc,5))+'_'+str(round(test_epoch_loss,5))+'_'+str(end)+".model")
    
    write_log('train_acc:'+str(acc)+'  train_epoch_loss:'+str(epoch_loss.item())+'  test_acc:'+str(test_acc)+'  test_epoch_loss:'+str(test_epoch_loss))
    write_log(str(acc)+'\t'+str(epoch_loss.item())+'\t'+str(test_acc)+'\t'+str(test_epoch_loss))

    write_log('__________________test end__________________')
    
    train_acc_l.append(acc)
    train_epoch_loss_l.append(epoch_loss.item())
    test_acc_l.append(test_acc)
    test_epoch_loss_l.append(test_epoch_loss)
    write_log("______________________________________________")
    write_log("______________________________________________")
    write_log("_______________train epoch "+str(epoch_num)+" end_______________")
    write_log("______________________________________________")
    write_log("______________________________________________")
    return min_test_epoch_loss











def test(device_test,test_loader,write_log,cls):
    cls.to(device_test)
    cls.eval()

    epoch_loss=0
    total=0
    correct=0
    output_all=[]
    label_all=[]
    for batch_idx,batch in enumerate(test_loader):
        with torch.no_grad():
#             print(batch['label'])
            label=batch['label'].to(device_test)#batch size * 1
            label_all.append(label.view(-1,1))
            input_ids=torch.stack(batch['input_ids']).t().to(device_test)#batch size * 100
            attention_mask=torch.stack(batch['attention_mask']).t().to(device_test)#batch size * 100
            
            #计算输出
            output = cls(input_ids, attention_mask=attention_mask)#batch size * 1
            total+=len(output)
            
            #计算loss
            
#             print(output,label)
            loss = criterion(output, label)
            epoch_loss+=loss
            ave_loss=epoch_loss/total
            
            #四舍五入
            output=softmax(output)
            output=output.argmax(dim=1)
            output_all.append(output)
            
            #计算准确率
            add_correct=(output== label).sum().item()
            correct+=add_correct
            acc=correct/total
            
            if batch_idx%5==0:
                print('[{}/{} ({:.0f}%)]\t正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    batch_idx, len(test_loader),100.*batch_idx/len(test_loader), 
                    correct, total,acc,
                    ave_loss
                    ),end= "\r")
            
            
            
    #结束：
    write_log('正确分类的样本数：{}，样本总数：{}，准确率：{}，ave_loss：{}'.format(
                    correct, total,acc,
                    ave_loss))
    
#     can't convert cuda:5 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    output_all=torch.cat(output_all,0)
    label_all=torch.cat(label_all,0)
    
    output_all=np.array(output_all.cpu())
    label_all=np.array(label_all.cpu())
    acc_score=metrics.accuracy_score(label_all,output_all)
    write_log(metrics.classification_report(label_all,output_all))
    write_log("准确率:"+str(acc_score))
    
    
    return acc,epoch_loss.item()

# test(device1)


