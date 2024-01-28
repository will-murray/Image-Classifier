import mnist_reader
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sys
from sklearn.neural_network._multilayer_perceptron import MLPClassifier





def pre_process(kind):
    X,y = mnist_reader.load_mnist(path= "data/fashion" , kind = kind)
    X = X.reshape(X.shape[0], 784)
    y = y.reshape(y.shape[0], )
    X = X/255
        
    indices_5 = np.where(y == 5)[0]
    indices_7 = np.where(y == 7)[0]
    if kind == "train":
        indices_5 = indices_5[:3000]
        indices_7 = indices_7[:3000]

    combined_indices = np.concatenate([indices_5,indices_7])
    np.random.shuffle(combined_indices)
    X = X[combined_indices]
    y = y[combined_indices]
    return [X,y]

def initialize_data():

    X,y = pre_process("train")
    X_test,y_test = pre_process("t10k")
    
    #flip the labels with probability of 0.2
    mask = np.random.rand(y.shape[0]) < 0.2
    y[mask & (y == 5)] = 7
    y[mask & (y == 7)] = 5


    return [X,y,X_test,y_test]


def cross_validation_error(model,X,y,k):

    N = X.shape[0]
    scores = []
    for i in range(0, k):
        start = int(i*np.floor(N/k))
        end = int((i+1)*np.floor(N/k) -1)
        val_indices = range(start,end)

        X_train = np.delete(X, val_indices, axis=0)
        y_train = np.delete(y, val_indices, axis=0)
        
        X_test = X[val_indices]
        y_test = y[val_indices]

        model.fit(X_train,y_train)
        scores.append(1- model.score(X_test,y_test) )

    return sum(scores) / k 


def regularization_test(
        kernel = "linear",
        samples = 10,
        gamma = 'auto',
        visualize = False,
        state = "tuning"
        ):

    assert(state in ["tuning", "testing"])

    X,y,X_test,y_test = initialize_data()

    C_values = [10**i for i in np.linspace(-2,2, samples)]

    errs = []
    test_errs = []
    idx = 1
    for C in C_values:
        model = SVC(kernel= kernel, C = C, gamma= gamma)
        if state == "tuning":
            errs.append(cross_validation_error(model, X,y,5))
        else:
            model.fit(X,y)
            errs.append(1 - model.score(X,y))
            test_errs.append(1 - model.score(X_test,y_test))
        print("{:.1f}%".format(idx/len(C_values) * 100))
        sys.stdout.write("\033[F")
        idx+=1


    min_C = min(zip(C_values, errs), key=lambda x: x[1])[0]
    print(f"{gamma} : {min_C}")
    if(visualize):
        if state == "tuning":
            plt.plot(C_values,errs, label = "Cross Validation Error", color = "green")
        elif state == "testing":
            plt.plot(C_values,errs, label = "Training Error", color = "green")
            plt.plot(C_values,test_errs, label = "Test Error", color = "purple")
            
        plt.xscale('log')
        plt.xlabel("Regularization Parameter C")
        plt.ylabel("Training Error")
        plt.title(f"gamma = {gamma}")
        plt.legend()
        plt.show()

    return(min(errs))



def SVM_gauss_kernel():


    bandwidths = [10**i for i in np.linspace(-1,0, 10)]
    C_values = []
    for b in bandwidths:
        C_values.append(regularization_test(kernel='rbf', gamma=b,state="tuning", samples=10))
    
    plt.plot(bandwidths, C_values, label = "Minimum Cross Validation Error", color = "red")
    plt.xlabel("gamma")
    plt.ylabel("optimal C value")
    plt.xscale('log')
    plt.legend()
    plt.show()

    print([C_values, bandwidths])


def tuned_SVMs():
    X,y,X_test,y_test = initialize_data()
    tuned_SVM_pairs = [
        [10.0, 10.0, 10.0, 10.0, 2.6826957952797246, 1.3894954943731375, 1.3894954943731375, 1.3894954943731375, 1.3894954943731375, 1.3894954943731375],
        [0.001, 0.0021544346900318843, 0.004641588833612777, 0.01, 0.021544346900318832, 0.046415888336127774, 0.1, 0.21544346900318823, 0.46415888336127775, 1.0]]
    gammas = tuned_SVM_pairs[1]
    C_values = tuned_SVM_pairs[0]
    
    train_errs = []
    test_errs = []
    idx = 1
    for (gamma, C) in zip(gammas, C_values):

        model = SVC(kernel="rbf",gamma=gamma, C=C)
        model.fit(X,y)
        train_errs.append(1-model.score(X,y))
        test_errs.append(1-model.score(X_test,y_test))

        print("{:.1f}%".format(idx/len(C_values) * 100))
        sys.stdout.write("\033[F")
        idx+=1

    

    min_test_index = test_errs.index(min(test_errs))

    min_test_gamma = gammas[min_test_index]
    min_test_err = test_errs[min_test_index]






    plt.plot(gammas,train_errs, label = "Training Error",color = 'green')
    plt.plot(gammas,test_errs, label = "Testing Error", color = 'purple')
    plt.scatter(min_test_gamma, min_test_err, label = f"minimum testing error : {min_test_err:.6f}",color = "darkgreen")
    plt.xscale('log')
    plt.legend()
    plt.xlabel("gamma")
    plt.xlabel("error")
    plt.show()

def neural_network_test():
    from sklearn.neural_network._multilayer_perceptron import MLPClassifier
    X,y,X_test,y_test = initialize_data()
    hidden_nodes = range(2,10)
    tanh_scores=[]
    logistic_scores = []
    relu_scores = []

    for h in hidden_nodes:
        print(f" hidden nodes= {h}")
        tanh_model = MLPClassifier(hidden_layer_sizes=(h,), solver='sgd',activation='tanh')
        tanh_model.fit(X,y)
        tanh_scores.append(1-tanh_model.score(X,y))

        logistic_model = MLPClassifier(hidden_layer_sizes=(h,), solver='sgd',activation='logistic')
        logistic_model.fit(X,y)
        logistic_scores.append(1-logistic_model.score(X_test,y_test))

        relu_model = MLPClassifier(hidden_layer_sizes=(h,), solver='sgd',activation='relu')
        relu_model.fit(X,y)
        relu_scores.append(1-relu_model.score(X_test,y_test))

    # plt.plot(hidden_nodes, testing_scores, color = "red", label = "testing error")
    plt.plot(hidden_nodes, tanh_scores, color = "green", label = "tanh Activation")
    plt.plot(hidden_nodes, logistic_scores, color = "purple", label = "logistic Activation")
    plt.plot(hidden_nodes, relu_scores, color = "orange", label = "relu Activation")

    plt.legend()
    plt.title("Multilayer Perceptron with 1 hidden layer")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Error")
    plt.show()

    
def hidden_nodes():
    X,y,X_test,y_test = initialize_data()


    hidden_nodes = [2,4,6,10,20,40, 80,160,320]
    errors = []
    test_errors = []
    idx = 1
    for h in hidden_nodes:
        model = MLPClassifier(hidden_layer_sizes= (h,), solver='adam',activation='relu', learning_rate='adaptive', max_iter=500, alpha=0.028,random_state=0)
        model.fit(X,y)
        errors.append(cross_validation_error(model,X,y,5))
        # test_errors.append(1-model.score(X_test,y_test))
        print("{:.1f}%".format(idx/len(hidden_nodes) * 100))
        sys.stdout.write("\033[F")
        idx+=1

    min_err_index = errors.index(min(errors))

    plt.plot(hidden_nodes, errors, color = "green", label = 'Training Error')
    # plt.plot(hidden_nodes, test_errors, color = "purple", label = 'Testing Error')
    plt.scatter(hidden_nodes[min_err_index], errors[min_err_index], label = f"min (nodes, test err) = ({hidden_nodes[min_err_index]:.0f},{errors[min_err_index]:.3f})" )
    
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Error")
    plt.xscale('log')
    plt.legend()
    plt.show()



def L2_test():
    from sklearn.neural_network._multilayer_perceptron import MLPClassifier
    X,y,X_test,y_test = initialize_data()



    alpha = [10**i for i in np.linspace(-2,2,10)] 
    errors = []
    idx = 1
    for a in alpha:
        model = MLPClassifier(hidden_layer_sizes= (6,), solver='adam',activation='relu', learning_rate='adaptive', max_iter=500, alpha=a)
        model.fit(X,y)
        errors.append(cross_validation_error(model,X,y,5))
        print("{:.1f}%".format(idx/len(alpha) * 100))
        sys.stdout.write("\033[F")
        idx+=1

    min_err_index = errors.index(min(errors))

        
    plt.plot(alpha, errors, color = "blue", label = 'cross validation error')
    plt.scatter(alpha[min_err_index], errors[min_err_index], label = f"min (alpha, err) = ({alpha[min_err_index]:.3f},{errors[min_err_index]:.3f})" )

    plt.xlabel("L2 regularization")
    plt.ylabel("Cross Validation Error")
    plt.xscale('log')
    plt.legend()
    plt.show()


def max_epocs_test():
    X,y,X_test,y_test = initialize_data()

    epocs = [1,10,25,50,75,100,250,500,1000,2500,5000]
    test_errs = []
    train_errs = []
    for e in epocs:
        print(e)
        model = MLPClassifier(hidden_layer_sizes= (24,), solver='adam',activation='relu', learning_rate='adaptive', max_iter=e, alpha=1, random_state=0)
        model.fit(X,y)
        train_errs.append(1-model.score(X,y))
        test_errs.append(1- model.score(X_test,y_test))

    plt.plot(epocs,train_errs,label = "Training Error",color = "green")
    plt.plot(epocs,test_errs,label = "Testing Error",color = "purple")
    plt.xlabel("Maximum Number of Epocs")
    plt.ylabel("Error")
    plt.xscale('log')
    plt.legend()

    plt.show()




def model_report(model,n,X_test,Y_test):
    testing_err = 1 - model.score(X_test,Y_test)
    lower = max((-1.36/np.sqrt(n)) + testing_err, 0)
    upper = (1.36/np.sqrt(n)) + testing_err
    print("testing err: ", testing_err)
    return [lower,upper, testing_err]


def plot_confidence_intervals(cis, labels):
    plt.figure(figsize=(10, 6))
    for index, (upper, lower, _) in enumerate(cis):
        plt.scatter([index], [(upper + lower)/2], color='r', s=100)  # Center error point
        plt.plot([index, index], [lower, upper], color='b', lw=7)  # Confidence interval line
    plt.xticks(ticks=range(len(cis)), labels=labels)
    plt.title("Confidence Intervals for Models")
    plt.ylabel("Error Rate")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def part4():
    X,y,X_test,y_test = initialize_data()

    linear_SVM = SVC(kernel='linear', C=0.026).fit(X,y)
    linear_CI = model_report(linear_SVM,2000,X_test,y_test)
    print(f"linear CI = [{linear_CI[0]} {linear_CI[1]}]")


    gaussian_SVM = SVC(kernel='rbf', gamma = 0.02154, C=2.68269).fit(X,y)
    gauss_CI = model_report(gaussian_SVM,2000,X_test,y_test)
    print(f"gaussian CI = [{gauss_CI[0]} {gauss_CI[1]}]")
    MLP = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',solver='adam', alpha=0.28, max_iter=500, learning_rate='adaptive').fit(X,y)
    MLP_CI = model_report(MLP,2000,X_test,y_test)
    print(f"MLP CI = [{MLP_CI[0]} {MLP_CI[1]}]")

    
    plot_confidence_intervals([linear_CI, gauss_CI, MLP_CI], ["Linear SVM", "Gaussian SVM", "MLP"])

# part4()


