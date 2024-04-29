% Fix d, alpha, tau
% Plot error against kappa on log scale

clear all;
close all;

d = 40;
tau = 10; P = floor(tau*d^2); % CHECK: Step size * ds^2 should be an integer
alpha=10; N = floor(alpha*d);

kappas = [0.1, 0.2,   0.25,  0.35,  0.45,  0.6,   0.75,  1,    1.3,   1.7, 2.2,   2.85 , 3.75  ,4.85 , 6.25 , 8.15, 10.55 ,13.65, 17.7,  22.95, 29.75, 38.55, 49.95];
Ks = floor(kappas*d);

sigma = 0.1; % std dev of noise
psi = 1; % sigma_beta
lambda = 0.01; % regularisation

numavg = 10; % Number of times to rerun experiment to smooth empirical plot noise
emps = zeros(1,length(Ks)); % empirical values for plots
for k=1:length(Ks)
    K = Ks(k);
    disp(k)
    for dummy = 1:numavg
        H = zeros(d^2,d^2);
        b = zeros(d^2,1);
        E = psi*randn(d, K);
        for mu = 1:P
            X = sqrt(1/d)*randn(d,N); % data ~ N(0,C/d) and C=I
            xNp1 = sqrt(1/d)*randn(d,1); % data ~ N(0,C/d) and C=I
            beta = E(:,randi(K));
            nue = sigma*randn(N,1);
            ymu = beta'*xNp1; %test y value

            Hmu = xNp1*((X*X'*beta+X*nue)');
            Hmu = reshape(Hmu', 1, numel(Hmu))';
            bmu = ymu*Hmu;

            H = H + Hmu*Hmu';
            b = b + bmu;
        end
        % compute Gamma as a matrix
        Gamma = (H + d*tau*lambda*eye(d^2))\b; % w correct scaling
        % OR ...
        % Gamma = pinv(H)*b

        Gamma = reshape(Gamma,[d,d]);
        emps(k) = emps(k) + diverseempiricalloss(Gamma,P,N,d,sigma,psi,30);
    end
    emps(k) = emps(k)/numavg;
end

figure()
hold on 
scatter(Ks/d,emps,30,"filled",'DisplayName','Empirical')
xline(1,'DisplayName','Transition')
xscale("log") 
legend