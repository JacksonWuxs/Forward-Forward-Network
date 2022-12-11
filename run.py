import numpy as np

from nn import MLPBinaryClassifier
from forwardforward import ForwardForwardClassifier
from base.utils import prepare_dataset, scoring

    
def pipeline(model, train, test):
    name = model.name
    trainX, trainY = train[:, :-1], train[:, -1]
    print("")
    print("Model: %s\n" % name + "-" * 40)
    model.fit(trainX, trainY)
    
    acc, f1, auc = scoring(trainY, model.predict(trainX))
    print("Train Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc, f1, auc))

    acc, f1, auc = scoring(test[:, -1].astype(np.int32), model.predict(test[:, :-1]))
    print("Test Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc, f1, auc))



if __name__ == "__main__":
    train, test, labels = prepare_dataset("./iris.csv", do_normalize=True)
    pipeline(ForwardForwardClassifier(in_dim=4, hide_dim=100, out_dim=2), train, test)
    pipeline(MLPBinaryClassifier(in_dim=4, hide_dim=100), train, test)
        
