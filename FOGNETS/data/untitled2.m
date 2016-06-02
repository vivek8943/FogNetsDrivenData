
Y_train = Train.Active_Customer;    
X_train = Train(:,1:255);                   % select predictor variables
X_test = Test(:,1:255);      




t = templateTree('Surrogate','on');
ens = fitensemble(X_train,Y_train,'AdaBoostM1',250,t);

predictions=ens.predict(X_test);