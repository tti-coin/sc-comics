# Copy *.txt files to 5-fold CV directories

import os

for fold in range(1,6):
    for i in range(1,1001):
        s = "{0:04d}".format(i)

        # for dev1,...,dev5
        if (i>(fold-1)*200 ) and ( i<=fold*200 ):
            print("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/" )
            os.system("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/" )

            # for dev1/1/,...,dev5/1/
            if (i>(fold-1)*200 ) and ( i<=(fold-1)*200+100 ):
                print("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/1/" )
                os.system("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/1/" )

            # for dev1/2/,...,dev5/2/
            else:
                print("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/2/" )
                os.system("cp -a "+str(s)+".txt 5-fold_CV/dev"+str(fold)+"/2/" )                    

        # for train1,...,train5
        else:
            print("cp -a "+str(s)+".txt 5-fold_CV/train"+str(fold)+"/" )
            os.system("cp -a "+str(s)+".txt 5-fold_CV/train"+str(fold)+"/" )
