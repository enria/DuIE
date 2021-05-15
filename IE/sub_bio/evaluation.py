def catch_dev0(a,b):
    if b==0:
        return 0
    return a/b


class F1Counter(object):
    def __init__(self):
        self.pred_cnt=0
        self.gold_cnt=0

        self.correct_pred=0
    
    def cal_score(self):
        precision=catch_dev0(self.correct_pred,self.pred_cnt)
        recall=catch_dev0(self.correct_pred,self.gold_cnt)
        f1=catch_dev0(2*precision*recall,precision+recall)
        return precision,recall,f1
    
    def __add__(self,b):
        result=F1Counter()
        result.pred_cnt=self.pred_cnt+b.pred_cnt
        result.gold_cnt=self.gold_cnt+b.gold_cnt
        result.correct_pred=self.correct_pred+b.correct_pred
        return result

def evaluate_pointer(pred,gold,counter=F1Counter()):
    for predi,goldi in zip(pred,gold):
        counter.pred_cnt+=(predi == 1).cpu().sum().item()
        counter.gold_cnt+=(goldi == 1).cpu().sum().item()
        counter.correct_pred+=((predi == 1) & (goldi.data == 1)).cpu().sum().item()
    return counter

def evaluate_span(pred,gold,counter=F1Counter()):
    for predi,goldi in zip(pred,gold):
        counter.pred_cnt+=len(predi)
        counter.gold_cnt+=len(set(goldi))
        counter.correct_pred+=len(set(predi)&set(goldi))
    return counter

def evaluate_concurrence(pred,gold,counter=F1Counter()):
    for predi,goldi in zip(pred,gold):
        counter.pred_cnt+=(predi == 1).cpu().sum().item()
        counter.gold_cnt+=(goldi == 1).cpu().sum().item()
        counter.correct_pred+=((predi == 1) & (goldi.data == 1)).cpu().sum().item()
    return counter
