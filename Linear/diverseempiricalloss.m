function result = diverseempiricalloss(Gamma, P, N, d, sigma, sigmabeta, numsamples)
    % Fully DIVERSE testing by sampling numsamples amount of times. 
   
    N = int32(N);
    avgerr = 0;
    for i = 1:numsamples
        onesampavg = 0;
        for j = 1:P
            X = sqrt(1/d)*randn(d,N); % data ~ N(0,C/d) and C=I
            xNp1 = sqrt(1/d)*randn(d,1); % data ~ N(0,C/d) and C=I
            beta = sigmabeta*randn(d, 1);
            nue = sigma*randn(N,1);
            ymu = beta'*xNp1; %test y value
            ypred = xNp1'*(Gamma*(X*X'*beta + X*nue));
            onesampavg = onesampavg + norm(ymu-ypred)^2;
        end
        avgerr = avgerr + onesampavg/P;
    end
    result = avgerr/numsamples;
end