# ---------- ENTIRE CODE ------------------------- #

library(caret) # package for fata splitting procedures, etc.
library(RSNNS) # package for Multilayer Perceptron
library(keras) # package for RNN

# ---------- Hyperparameters --------------------- #
seed=1001
epochs_MLP=1000
epochs_RNN=100
hidden_MLP=34
hidden_RNN=55

# ---------- Data Importing ----------------------- #
setwd("enter_path_here")
#setwd("/Users/artemvysogorets/Riemann_Zeta_NN")
data=as.data.frame(read.csv("data_features_50.csv"),header=TRUE)

# ---------- Save unscaled data ------------------- #
data_us=data

# ---------- Scaling ------------------------------ #
max_Y=max(data[,1])
min_Y=min(data[,1])
for(i in 1:ncol(data))
{data[,i]=(data[,i]-min(data[,i]))/(max(data[,i])-min(data[,i]))}

# ---------- Data splitting (RRN ONLY) ------------ #
train_nbr=floor(0.8*nrow(data))
train_set=data[(1:train_nbr),]
train_set_us=data_us[(1:train_nbr),]
train_X=as.matrix(train_set[,-1])
train_Y=as.matrix(train_set[,1])
train_X_us=as.matrix(train_set_us[,-1])
train_Y_us=as.matrix(train_set_us[,1])
test_set=data[-(1:train_nbr),]
test_set_us=data_us[-(1:train_nbr),]
test_X=as.matrix(test_set[,-1])
test_Y=as.matrix(test_set[,1])
test_X_us=as.matrix(test_set_us[,-1])
test_Y_us=as.matrix(test_set_us[,1])

# ---------- Model Construction (RNN ONLY) ----------#
set.seed(seed)
model=keras_model_sequential()
layer_dense(model,units=hidden_RNN, activation='relu', input_shape=c(ncol(train_X)))
layer_dense(model,units=hidden_RNN, activation='relu', input_shape=c(ncol(train_X)))
layer_dense(model,units=hidden_RNN, activation='relu', input_shape=c(ncol(train_X)))
layer_dense(model,units=hidden_RNN, activation='relu', input_shape=c(ncol(train_X)))
layer_dense(model,units=1, activation='sigmoid')
compile(model,loss='mean_squared_error',optimizer='adam')

# ---------- Training (RNN ONLY) -------------------- #
set.seed(seed)
fit(model,x=train_X,y=train_Y,epochs=epochs_RNN,batch_size=32, view_metrics=TRUE)

# ---------- Replicative accuracy (RNN only) ------- #
ps_train=predict(model, train_X)
ps_train=unlist(ps_train)
ps_train_us=(ps_train*(max_Y-min_Y)+min_Y)
RMSE_train_us=((1/nrow(train_set_us))*sum((train_Y_us-ps_train_us)^2))^(1/2)
R_train_us=1-(sum((train_Y_us-ps_train_us)^2))/(length(train_Y_us)*var(train_Y_us))

# ---------- Predictive accuracy (RNN ONLY) -------- #
ps_test=predict(model,test_X)
ps_test=unlist(ps_test)
ps_test_us=(ps_test*(max_Y-min_Y))+min_Y
RMSE_test_us=((1/nrow(test_set_us))*sum((test_Y_us-ps_test_us)^2))^(1/2)
R_test_us=1-(sum((test_Y_us-ps_test_us)^2))/(length(test_Y_us)*var(test_Y_us))

# ----------- Graphing (RNN ONLY) ------------------ #
par(mar=c(5,6,4,1)+1)
plot(c(1:50),test_Y_us[50:99], type="o", pch=3, 
xlab="\n Index", ylab = "Distance \n" ,family="serif", 
cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
lines(ps_test_us[50:99], type="o", col=28, pch=8)
op=par(family = "serif")
legend("bottomright", legend=c("Predicted", "Actual"), pch=c(3,8), 
col = c("black", 28), cex=1.5, pt.cex = 1, bty="n", 
x.intersp=0.15, text.width = 5)
par(op)

# ----------- Data Splitting (MLP ONLY) ------------ #
set.seed(seed)
data=data[sample(nrow(data)),]
set.seed(seed)
data_us=data_us[sample(nrow(data)),]
set.seed(seed)
train_vec=createDataPartition(data[,1],p=0.8,list=FALSE)
train_set=data[train_vec,]
test_set=data[-train_vec,]
train_set_us=data_us[train_vec,]
test_set_us=data_us[-train_vec,]

# ----------- Training (MLP ONLY) ------------------ #
set.seed(seed)
model=mlp(x=train_set[,-1],y=train_set[,1],size=c(hidden_MLP,hidden_MLP),maxit=epochs_MLP)

# ----------- Replicative Accuracy (MLP ONLY) ------ #
ps_train=predict(model, train_set[,-1])
ps_train_us=(ps_train*(max_Y-min_Y))+min_Y
R_train_us=1-(sum((train_set_us[,1]-ps_train_us)^2))/(length(train_set_us[,1])*var(train_set_us[,1]))
RMSE_train_us=((1/nrow(train_set_us))*sum((train_set_us[,1]-ps_train_us)^2))^(1/2)

# ----------- Predictive Accuracy (MLP ONLY) ------- #
ps_test=predict(model, test_set[,-1])
ps_test_us=(ps_test*(max_Y-min_Y))+min_Y
R_test_us=1-(sum((test_set_us[,1]-ps_test_us)^2))/(length(test_set_us[,1])*var(test_set_us[,1]))
RMSE_test_us=((1/nrow(test_set_us))*sum((test_set_us[,1]-ps_test_us)^2))^(1/2)

# ----------- Grpahing (MLP ONLY) ------------------ #
par(mar=c(5,6,4,1)+1)
plot(c(1:50),test_set_us[49:98,1], type="o", pch=3, 
xlab="\n Index", ylab="Distance \n" ,family="serif", 
cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
lines(ps_test_us[49:98], type="o", col=28, pch=8)
op=par(family="serif")
legend("topright", legend=c("Predicted", "Actual"), pch=c(3,8), 
col=c("black", 28), cex=1.5, pt.cex=1, bty="n", 
x.intersp=0.25, text.width = 5)
par(op)

# ----------- Printing Errors ---------------------- #
paste("Replicative accuracy: RMSE of ", RMSE_train_us,sep="")
paste("Replicative accuracy: R of ", R_train_us,sep="")
paste("Predictive accuracy: RMSE of ", RMSE_test_us,sep="")
paste("Predictive accuracy: R of ", R_test_us,sep="")

# ----------- COMMENTS ----------------------------- #
# "_us" suffix in predictions and error calculation stands for "unscaled".
# For example, RMSE_test_us refers to RMSE score on unscaled test data.

