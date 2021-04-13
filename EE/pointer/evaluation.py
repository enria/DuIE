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

def evaluate(pred,gold,counter=F1Counter()):
    for predi,goldi in zip(pred,gold):
        counter.pred_cnt+=len(predi)
        counter.gold_cnt+=len(goldi)
        counter.correct_pred+=len(set(predi)&set(goldi))
    return counter
