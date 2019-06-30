from getting_trainingData import predict
try:
    import dill as pickle
except ImportError:
    import pickle



model1 = pickle.load(open('model.pkl','rb'))
# making the predictions
def makePred():
    return predict('file.wav', model1)

print(makePred())