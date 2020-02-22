from sklearn.metrics import accuracy_score
import numpy as np
import copy

class SDRClassifier:
    """
    SDR Classification
    """
    def __init__(self, p=0.5 , msc=-1, loop_limit=3):
        self.p = p
        self.msc = msc
        self.loop_limit = loop_limit
        self.candidate_rules = []
        self.rules = []
        
    #Get unique values 
    def __UniqueLabelCounter(self,  df,  colname=None):
        if colname == None:
             return np.unique(df)     
        return np.unique(df[colname])
    
    #get unique value frequencies for  each class
    def __LabelCounterByUniqueValue(self,  X,  y,  val_colname):
        result = {}
        result2 = {}
        values = self.__UniqueLabelCounter(df=X[val_colname]) #get col uniques
        labels = self.__UniqueLabelCounter(df=y) #get classes
        
        for i in values:
            result2.clear()
            
            for t in labels: #initilze
                result2[t]=0
            for j in range(0, len(y)):
                if X[val_colname][j] == i :
                    lbl = y[j]
                    result2[lbl] = result2[lbl] + 1
            
            result[i] = result2.copy()
        return result
    
    #candidate rule creator
    def __createCandidateRules(self, resultdict, col):
        result = {}
        values = list(resultdict.keys())
        labels = list(resultdict[values[0]].keys())

        #calculation min max weighted avg    
        for i in labels: #each class
            sum,  counter = 0,  0
            items = []
            for j in values:                  
                item = resultdict[j]
                sum = sum + (item[i]*j)     
                counter = counter + item[i]
                if item[i] != 0:
                    items.append(j)

            if len(items) == 0: 
                continue

            result["Min"],  result["Max"],  result["Avg"] = min(items),  max(items),  (sum/counter)
            
            #creating candidate rules
            if result["Min"] != result["Max"]:
                #print(" => ", result2["Min"], " <= ", col, " <= ", result2["Avg"])
                #print(" => ", result2["Avg"], " <= ", col, " <= ", result2["Max"])
                self.candidate_rules.append(self.__rule(minn = result["Min"], 
                                                 maxx = result["Avg"], 
                                                 feature = col, 
                                                 label = None, 
                                                 rate = None, 
                                                 sc = None))
                self.candidate_rules.append(self.__rule(minn = result["Avg"], 
                                                 maxx = result["Max"], 
                                                 feature = col, 
                                                 label = None, 
                                                 rate = None, 
                                                 sc = None))
            #min value and  max value same
            else:
                #print(" => ", result2["Min"], " <= ", col, " <= ", result2["Max"])
                self.candidate_rules.append(self.__rule(minn = result["Min"], 
                                                 maxx = result["Max"], 
                                                 feature = col, 
                                                 label = None, 
                                                 rate = None, 
                                                 sc = None ))

    #voting process  
    def __Voting(self,  X,  y,  colname): 
        labels = self.__UniqueLabelCounter(y)
               
        for rule in self.candidate_rules:
            filt = (X[colname] >= rule.minn) & (X[colname] <= rule.maxx)
            df = X[filt] #filter, it finds instances that matched with candidate rule
            
            dic = {}
            for i in labels: #init dic
                dic[i] = 0
                
            for i in df.index:
                lbl = y[i]
                dic[lbl] = dic[lbl] + 1
            
            total = sum(dic.values())
            best = max(dic.values())
            k = [k for k, v in dic.items() if v==best]
            rate = (best/total)
        
            if  rate >= self.p and total >= self.msc:    #is it reliable rule ?
                
                #update rule parameters
                rule.label = k
                rule.rate = rate
                rule.sc = total
                
                self.rules.append(rule)
                #print(rule.minn, "  ", rule.maxx, "  ", rule.rate, "  ", total)
            
        self.candidate_rules.clear() #clear candidates rules

    #filter dateset               
    def __filter_databyRules(self,  X,  y):
        for i in self.rules:
            col = i.feature
            filter = (X[col] < i.minn) | (X[col]>i.maxx)
            X = X[filter]
            y = y[filter]

        return X,  y   
      

    def fit(self,  X_train,  y_train):
        iteration = 0
        last_rules_count = -1 # for redundent detection

        cols = list(X_train.columns.values) #get columns

        while True:
            iteration = iteration + 1 
            X_train.reset_index(drop=True,  inplace=True)
            y_train.reset_index(drop=True,  inplace=True)

            for i in cols: #Creating and voting candidate rules each features
                rslt = self.__LabelCounterByUniqueValue(X=X_train, y=y_train, val_colname=i)
                self.__createCandidateRules(rslt, i)
                self.__Voting(X = X_train,  y = y_train,  colname = i)

            
            if last_rules_count == len(self.rules): #is this redundent iteration ?
                    #print(len(self.rules))
                    self.loop_limit = self.loop_limit -1
                    if self.loop_limit == 0: #terminate state1
                        break
            else:
                last_rules_count = len(self.rules)

            #filtering dataset with selected rules
            X_train,  y_train = self.__filter_databyRules(X=X_train,  y=y_train)

            if len(X_train)==0: #terminate state2
                break 

         
    
    def __sort(self):
        sort = []
        rl = self.rules.copy()
        while rl != []:
            k = [i.sc for i in rl]
            best = max(k)
            for i in self.rules:
                if i.sc == best:
                    sort.append(i)
                    rl.remove(i)
        return sort
                           
    def predict(self,  X_test):
        
        self.rules = self.__sort()
        pred = np.array([])
        for j in range(len(X_test)):
            sample = dict(X_test.iloc[j, :])              #get test sample
            ignore = True
            for i in self.rules:                              
                rslt = i.check(dic=sample)                #if sample match with rule
                if rslt == "-":                           #rule not matched
                    continue
                else:
                    ignore = False
                    pred = np.append(pred, rslt)         
                    break
            if ignore:
                pred = np.append(pred, ["-1"])        
        return pred
      
    def showRules(self):
        for i in self.rules:
            print("F:", i.feature, "Min :", i.minn, " - Max :", i.maxx, " - Label :", i.label," - Rate :", i.rate, "Sampling Size :",i.sc)
 

    #for the find best p and msc params
    def optimizeParameters(self,  X_train,  y_train,  X_test,  y_test):
        p_vals = np.arange(0.5,  1.0,  0.01)
        #print(p_vals)
        best_acc,  best_p,  best_msc,  best_SDR   = -1,  -1,  -1,  None
        
        #p optimizing
        print("P optimizing...")
        for i in p_vals:
            i = round(i,3)
            SDR = SDRClassifier(p=i)
            SDR.fit(X_train=X_train,  y_train=y_train)
            pred = SDR.predict(X_test=X_test)
            acc = accuracy_score(y_test,  pred)
            #print("p :",  i,  " acc :",  acc,  flush = True)
            
            if acc > best_acc:
                best_SDR = SDR
                best_acc = acc
                best_p = i
                print("Improved! - Acc:", acc, " - p:", i)#, flush=True
        print("Done.")
        #print("Acc:", best_acc, " ve p:", best_p, " ile devam et..", flush=True)
        #msc optimizing
        print("MSC optimizing...")
        while True: 
            SDR_ex = copy.deepcopy(best_SDR)
            SamplingSizes = [i.sc for i in SDR_ex.rules]
            if SamplingSizes == []:
                break
            msc = min(SamplingSizes)
            rules = [i for i in SDR_ex.rules if i.sc > msc]
            SDR_ex.rules = rules
            pred = SDR_ex.predict(X_test=X_test)
            acc = accuracy_score(y_test, pred)
            
            if acc < best_acc:
                break
            else:
                best_msc = msc
                best_SDR = SDR_ex
        
        print("Done.")
        print("Best :=> p:", round(best_p, 4), " msc:", best_msc, flush = True)        
        return best_p,  best_msc,  best_SDR
     
    #cross validation for sdr classification 
    def cross_validate(self,  X,  y,  fold):

        accs = []
        
        start_index = 0
        lenght = len(y)
        part_size = round(lenght/fold)
         
        print("len:", lenght)
        print("part", part_size)
            
        for i in range(1, fold + 1): #ikinci değer dahil olmadığı için +1
            
            end_index = start_index + part_size
            if end_index > lenght:
                end_index = lenght

            print("Start:",start_index," End:",end_index)

            #set train data
            X_train = X.drop(X.index[start_index:end_index], axis=0).reset_index(drop=True)
            y_train = y.drop(y.index[start_index:end_index], axis=0).reset_index(drop=True)
            
            #set test data
            X_test = X.iloc[start_index:end_index].reset_index(drop=True)
            y_test = y.iloc[start_index:end_index].reset_index(drop=True)
            print(y_test.unique(),flush=True)

            #optimize parameters
            best_params = self.optimizeParameters(X_train=X_train,  y_train=y_train,  X_test=X_test,  y_test=y_test)
            
            #accuracy
            pred =  best_params[2].predict(X_test=X_test, y_test=y_test)
            acc = accuracy_score(y_test, pred)
            accs.append(acc)
            print(i, ". SDR Classifier Cross Acc:", acc, flush=True)

            start_index = end_index
        
        print("Accs: ", accs, "Avg: ", sum(accs)/len(accs), flush=True)

    class __rule():
        """
        Rule Class
        param rate    : majority class rate
        param label   : rule label
        param feature : feature
        param minn    : min value
        param maxx    : max value
        """
        def __init__(self, minn, maxx, feature, label, rate, sc):
            self.minn = minn
            self.maxx = maxx
            self.feature = feature
            self.label = label
            self.rate = rate
            self.sc = sc

        #Check matching process
        def check(self,  dic):
            val = dic[self.feature]
            if self.minn <= val <= self.maxx:
                return self.label
            return "-"  