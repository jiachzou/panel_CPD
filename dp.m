% dp.m
% This program implements the dynamic programming algorithm
% for determining the three change times of a signal that
% consists of four unknown DC levels in WGN (see Figure 12.4).
% The DC levels and change times are unknown.
%
clear all
% Generate data
randn('seed' ,0)
A=[1;4;2;6] ;varw=1;sig=sqrt(varw);
x=[sig*randn(20,1)+A(1); sig*randn(30,1)+A(2); ...
    sig*randn(15,1)+A(3);sig*randn(35,1)+A(4)] ;
N=length(x);
% Begin DP algorithm
% Since MATLAB cannot accommodate matrix/vector indices of zero,
% augment L,k,n by one when necessary.
% Initialize DP algorithm
for L=0:N-4
    LL=L+1 ;
    I(1,LL)=(x(1:LL)-mean(x(1:LL)))'*(x(1:LL)-mean(x(1:LL)));
end

%%

% Begin DP recursions
for k=1:3
    kk=k+1;

    if k<3
    for L=k:N-4+k
    LL=L+1 ;
   
% Load in large number to prevent minimizing value of J
% to occur for a value of J(l:k), which is not computed
    J(1:k)=10000*ones(k,1);
 
    %%

% Compute least squares error for all possible change times
    for n=k:L
    nn=n+1 ;
    Del=(x(nn:LL)-mean(x(nn:LL)))'*(x(nn:LL)-mean(x(nn:LL)));
    J(nn)=I(kk-1,nn-1)+Del;
end
% Determine minimum of least squares error and change time that
% yields the minimum
[I(kk,LL),ntrans(L,k)]=min(J(1:LL));
end
else
% Final stage computation
L=N-1;LL=L+1;J(1:k)=10000*ones(k,1);
for n=k:N-1
    nn=n+1 ;
    Del=(x(nn:LL)-mean(x(nn:LL)))'*(x(nn:LL)-mean(x(nn:LL)));
    J(nn)=I(kk-1,nn-1)+Del;
    end
    [Imin,ntrans(N-1,k)]=min(J(1:N)) ;
end
end
% Determine change times that minimize least squares error
n2est=ntrans(N-1,3);
nlest=ntrans(n2est-1,2);
nOest=ntrans(nlest-1,1);
% Reference change times to [O,N-l] interval instead of
% MATLAB's [l,N]
nOest=nOest-1
nlest=nlest-1
n2est=n2est-1