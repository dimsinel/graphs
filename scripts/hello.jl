using DrWatson
@quickactivate "graphs"


# Load the module and generate the functions
module CppHello
using CxxWrap
@wrapmodule(() -> joinpath("/home/lenny/Work/libcxxwrap-julia/build/lib", "libhello"))

function __init__()
    @initcxx
end
end


module Basic_Types
using CxxWrap
@wrapmodule(() -> joinpath("/home/lenny/Work/libcxxwrap-julia/build/lib", "libbasic_types"))

function __init__()
    @initcxx
end
end
# Call greet and show the result

@show CppHello.greet()

CppHello.add(1.1,2)


using MLUtils

# X is a matrix of floats
# Y is a vector of strings
X, Y = load_iris()

# The iris dataset is ordered according to their labels,
# which means that we should shuffle the dataset before
# partitioning it into training- and test-set.
Xs, Ys = shuffleobs((X, Y))

# We leave out 15 % of the data for testing
cv_data, test_data = splitobs((Xs, Ys); at=0.85)

# Next we partition the data using a 10-fold scheme.
for (train_data, val_data) in kfolds(cv_data; k=10)

    # We apply a lazy transform for data augmentation
    train_data = mapobs(xy -> (xy[1] .+ 0.1 .* randn.(), xy[2]),  train_data)

    for epoch = 1:10
        # Iterate over the data using mini-batches of 5 observations each
        for (x, y) in eachobs(train_data, batchsize=5)
            @show x, y
            # ... train supervised model on minibatches here
        end
    end
end

using MLJ
measures("FScore")
m1 = FScore()
m2 = MulticlassFScore()

y₀ = coerce(categorical(rand("abc", 10)), Multiclass)
#y₀ = categorical(rand("abc", 10))
y = coerce(categorical(rand("abc", 10) ), Multiclass) # or `coerce(rand("abc", 10), Multiclass))`
#ŷ = categorical(rand("abc", 10))
ŷ = coerce(categorical(rand("abc", 10) ), Multiclass)  # or `coerce(rand("abc", 10), Multiclass))`

MulticlassFScore()(ŷ, y)
#MulticlassFScore()(ŷ, y, class_w)



m1(y, ŷ) 
m1(y₀, ŷ)

m2(y, ŷ)
m2(y₀, ŷ)

cm = ConfusionMatrix()(ŷ, y)  # or `confmat((ŷ, y)`.

## scikit-learn example 
## at 
## http://tinyurl.com/yrc2vxds
#= 
import numpy as np
from sklearn.metrics import f1_score =#
y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 0, 1]
levels(y_true)
#=
f1_score(y_true, y_pred, average='macro')  ---> 0.26666666666666666
f1_score(y_true, y_pred, average='micro') ---> 0.3333333333333333
f1_score(y_true, y_pred, average='weighted')---> 0.26666666666666666
f1_score(y_true, y_pred, average=None)---> array([0.8, 0. , 0. ])
=#
m1(y_true, y_pred)
m2(y_true, y_pred)
# but MulticlassFscore has 3 aliases.
micro_f1score(y_true, y_pred)
macro_f1score(y_true, y_pred)
multiclass_f1score(y_true, y_pred)
confmat(y_true, y_pred)


using PyCall
using StatsPlots

py"""
import numpy as np
def compute_f1_score(hyperedge, predicted_hyperedge):
    common_part = np.bitwise_and(hyperedge, predicted_hyperedge)
    #print('--- >  ', common_part, hyperedge, predicted_hyperedge)
    f1_score = 0

    true_positives = sum(common_part)

    if true_positives > 0:
        #true_positives = len(common_part)
        total_positives = sum(predicted_hyperedge)
        actual_positives = sum(hyperedge)

        precision = true_positives / total_positives
        recall = true_positives / actual_positives

        f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score
"""

py"compute_f1_score"(y_true, y_pred)
# this does not agree w/ any of the above


### Another try 

microf1 = Vector{Float64}(undef,1)
macrof1 = Vector{Float64}(undef,1)
pyf1 = Vector{Float64}(undef,1)

for i in 1:10000
    y_true = rand([0, 1], 30)
    y_pred = rand([0, 1], 30)
    
    # confmat(y_true, y_pred)
    pyf = py"compute_f1_score"(y_true,y_pred)
    jmacrof = macro_f1score(y_true, y_pred)
    #jmicrof = m1(y_true, y_pred)
    jmicrof = micro_f1score(y_true, y_pred)
    #println("py $(pyf),  macro = $(jmacrof),  micro = $(jmicrof)")
    # m1(y_true, y_pred)
    # m2(y_true, y_pred)

    # but MulticlassFscore has 3 aliases.
    #micro_f1score(y_true, y_pred)
    push!( microf1, jmicrof )
    push!( macrof1, jmacrof )
    #multiclass_f1score(y_true, y_pred)
    #confmat(y_true, y_pred)
    push!( pyf1, pyf )
end

histogram(macrof1, label="macro") #, bins = .2:.01:.3)
histogram!(pyf1, label = "python") #, bins = .2:.01:.3)
h_micro = histogram!(microf1, label= "micro")

import StatsBase, Distributions
length(pyf1), typeof(pyf1)
h_microfit = StatsBase.fit(Distributions.Normal, pyf1)

# this must be a pdf or something, look it up
plot!(h_microfit, linewidth = 4)

a=[ 1 2 3; 4 5 6 ]
a[1,2], a[2,1]


clf = ConstantClassifier()
X, y = @load_crabs # a table and a categorical vector
mach = machine(clf, X, y) |> fit!

fitted_params(mach)
Xnew = (; FL=[8.1, 24.8, 7.2],
    RW=[5.1, 25.7, 6.4],
    CL=[15.9, 46.7, 14.3],
    CW=[18.7, 59.7, 12.2],
    BD=[6.2, 23.6, 8.4],)

yhat = predict(mach, Xnew)
yhat[1]

# raw probabilities:
pdf.(yhat, "B")

# probability matrix:
L = levels(y)
pdf(yhat, L)

# point predictions:
predict_mode(mach, Xnew)

?add_hyperedge



bit(x,y) = @pipe (x & x) |>  (digits(_, base=2, pad=8) |> reverse)'
