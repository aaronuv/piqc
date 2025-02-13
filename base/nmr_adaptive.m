% open-loop control with piece-wise constant controls for NMR
% n >= 2
clearup=1;

if clearup==1
    clear all;
    clearup=1;
end

rng('default');

%=======================================================================
% pauli matrices defs and storage
%=======================================================================

n=4;        % number of qubits
%NQBIT = 'two qubits';
NQBIT = 'n>2';

SMOOTHING = 'window';
%SMOOTHING = 'spline';
sp=0.8;    % cubic spline smoothing with 0 <= p <= 1. p=0 is straight line, p=1 is original data

NMR = n;

N=2^n;

B = build_B(n, NQBIT);

e{1} = [1 0;0 1];            %identity
e{2} = [0 1;1 0];            %sigma x
e{3} = [0 -1i;1i 0];         %sigma y
e{4} = [1 0;0 -1];           %sigma z

S = build_S(n);

switch NMR

    case 2
    % 2-qubit molecule
    M = 1e6;
    scale = 1e-3;
    params = scale*pi*[400*M 0; 47.6 376*M];

    case 4
    %4-qubit molecule
    scale=1e-3;
    params=scale*pi*[
         15479.88,        0,           0,         0;
          -297.71, -3132.45,           0,         0;
          -275.56,    64.74,   -42682.97,         0;
            39.17,     51.5,     -129.08, -56445.71
        ];


    case 7
    % 7-qubits
    scale=1e-3;
    params=pi*scale*[
        1750.3  0.      0.      0.      0.      0.      0.      ;
        40.800  14930.1 0.      0.      0.      0.      0.      ;
        1.6000  69.5000 12199.9 0.      0.      0.      0.      ;
        8.4700  1.40000 71.0400 17173.7 0.      0.      0.      ;
        4.0000  155.600 -1.8000 6.50000 2785.85 0.      0.      ;
        6.6400  -0.7000 162.900 3.30000 15.8100 2320.25 0.      ;
        128.00  -7.1000 6.60000 -0.9000 6.90000 -1.7000 718.487
    ];

    case 10
        scale=1e-3;
        params = ones(10, 10);

end

h2=0;
for i=1:n
    for j=1:i-1
        h2 = h2 + params(i, j)*S{3, 3, j, i};
    end
end

H0 = sparse(0.5*h2);

%=======================================================================

T=8.8;					% horizon time
%D=0.01*(scale*0.84);	% diffusion for 2-NMR
%D=1*(scale*1.64);	    % diffusion for 4-NMR
D=1*(scale*0.23);	    % diffusion for 4-NMR
%D=1e-5;
%r=8;
r=1;
nc=2*n;
beta=200;
R=r/nc;
n_traj=400;
nIS=1000;		% number of important sampling steps
NT=100;
dt=T/NT;

Q = beta*R;

Dtilde=D/2;     % unraveling diffusion

lambda=Dtilde*R;


psi0_oneq = [1,0]';    % single qubit initial state 1/sqrt(2)*(0 + 1)

psi0_oneq = psi0_oneq/norm(psi0_oneq);
psi0 = psi0_oneq;
for j=1:n-1
    psi0=kron(psi0_oneq, psi0);   % pis0 = 1q0 x 1q0 x ... x 1q0
end

pi_sumshifts = sum(diag(params))/scale;
phase = exp(1i*mod(pi_sumshifts*T, 2*pi));

phi=zeros(N,1);
phi(1)=phase/sqrt(2);
phi(N)=phase'/sqrt(2);

% end definitions -------------------------------------------------------

ssp=zeros(1, nIS);
Jp=zeros(1, nIS);
F_allp=zeros(1, nIS);
F_min_allp=zeros(1, nIS);
u_normp=zeros(1, nIS);
u_allp=zeros(nIS,2,n,NT);
von_neumann=zeros(n_traj, NT);
fid=zeros(n_traj, NT);
m_all=zeros(NT,n_traj,3);

S_path = zeros(nIS);


if clearup==1
    u=zeros(2,n,NT);			    % controls of single qubits (2  for (x, y), n qubits, NT time steps)
    u_av=zeros(2,n,NT);
end

%=======================

Dfinal=1e-12;
%Dinit=pi^2/T;
%Dinit=0.01;
Dinit=1e-6;


%DSCHEDULE='linear';
%DSCHEDULE='1/j';
%DSCHEDULE='1/j^2';
DSCHEDULE='exp';
%DSCHEDULE='const + exp';
%DSCHEDULE='steps';
%DSCHEDULE='logsteps';
%DSCHEDULE='ct';


SPLINE_SCHEDULE = 'linear';

switch SPLINE_SCHEDULE
    case 'linear'
        a = (1-0.1)/(nIS-1);
        b = Dinit-a;
        spall = a * (1:nIS) + b;
end

switch DSCHEDULE
case 'ct'
    Dtilde=D/2;                 % unraveling diffusion
    Dall=Dtilde*ones(nIS, 1);

case '1/j'
    a=1/(nIS-1)*(1/Dfinal-1/Dinit);
    b=1/Dinit-a;
    Dall = 1./(a*(1:nIS) + b);

case 'linear'
    a = (Dfinal-Dinit)/(nIS-1);
    b = Dinit-a;
    Dall = a * (1:nIS) + b;

case '1/j^2'
    a=1/(nIS-1)*(1/sqrt(Dfinal)-1/sqrt(Dinit));
    b=1/sqrt(Dinit)-a;
    Dall = 1./(a*(1:nIS) + b).^2;

case 'exp'
    a = (log(Dfinal)-log(Dinit))/(nIS-1);
    b = log(Dinit)-a;
    Dall = exp(a * (1:nIS) + b);

case 'steps'
    nsteps=5;
    lengthstep=nIS/nsteps;
    for k=1:nsteps
        Dall((k-1)*lengthstep + (1:lengthstep))=Dinit*(Dfinal/Dinit)^(k/nsteps);
    end

case 'logsteps' %WARNING: experimental
    nsteps=5;
    length = nIS;
    end_idx = 1;
    for k=1:nsteps
        start_idx = end_idx;
        end_idx = end_idx + ceil(length/2);
        length = ceil(length/2);
        Dall(start_idx:end_idx) = Dinit*(Dfinal/Dinit)^((k-1)/(nsteps-1));

    end

end


%=======================

K=50;
is_window=1;
jw= 0;
j0 = nIS;


outer_time=tic;
for j=1:nIS

    if j > j0
        jw = j0;
        is_window=1;
    end

    D=Dall(j);
    Dtilde=D/2;                 % unraveling diffusion

    lambda=Dtilde*R;


    inner_time=tic;
    u(:)=u_av(:);

    % initialize 
    psi=psi0*ones(1,n_traj);
%    m=m0'*ones(1,n_traj);
	% generate trajectories 
	dW=sqrt(Dtilde*dt)*randn(n_traj,2,n,NT);	% stores all noise realizations

    S_path = 0;
    % psi dynamics
    for t=1:NT

        dwu=reshape(u(:,:,t), [1 2 n])*dt + squeeze(dW(:,:,:,t));  % v=(n_traj, n_c, n)
        
        klad=squeeze(dwu(:,1,1))'.*(B{1,2,1}*psi) + squeeze(dwu(:,2,1))'.*(B{1,3,1}*psi); % 2=x 3=y
        klad1=klad;
        for p=1:n-1
            klad = klad + dwu(:,1,p+1)'.*(B{p,1,2}*psi) + dwu(:,2,p+1)'.*(B{p,1,3}*psi);
        end
        psi=psi -1i*H0*psi*dt -n*Dtilde*psi*dt -1i*klad;
        norm_psi = sum(psi.*conj(psi),1);
        psi=psi./(ones(N,1)*sqrt(norm_psi));
        time_fid=(abs(psi'*phi)).^2;
        fid(:, t) = time_fid;
    end

    S_path = -Q/2 * S_path * dt;
    Su=0.5*R*dt*sum(u.^2,"all");
    u_normp(j)=2/R*Su;
    S=Su + R*sum(dW.*reshape(u, [1 2 n NT]),[2 3 4]);
    F = (abs(psi'*phi)).^2;
    Send= Q/2*log(1 - F);
    %Send= -Q/2*F;
    F_allp(j)=mean(F);
    F_min_allp(j) =min(F);

    fprintf('IS=%2.f%%, F: %f %f \n', (j/nIS)*100, F_allp(j), std(F));
    S=S+Send;
    %S=S+S_path+Send;
    
    Smin=min(S);
	S=S-Smin;
	w=exp(-S/lambda);

	Jp(j)=Smin-lambda*log(mean(w));
	w=w/sum(w);
	ssp(j)=1/sum(w.^2)/n_traj;

    if mod(j, 40)==0
        figure(1)
        subplot(2,2,1)
        plot(ssp)
        ylabel('ss')
        subplot(2,2,2)
        plot(Jp)
        ylabel('J')
        subplot(2,2,3)
        plot(u_normp)
        ylabel('norm u')
        subplot(2,2,4)
        semilogy(1:nIS,1-F_allp,1:nIS,1-F_min_allp)
        ylabel('1-F')
        drawnow

        figure(2)
        for a=1:n
        subplot(n,1,a)
        plot(dt*(1:NT),reshape(u_av(:,a,:), [2, NT])')
        end

        drawnow
    end


    % K equal time intervals
    % NT is divisible by K
    N2=floor(NT/K);
    for k=1:K
        iv=((k-1)*N2+1):(k*N2);
        klad=1/dt/N2*tensorprod(w, sum(dW(:,:,:,iv), 4), 1, 1);
        u(:,:,iv)=u(:,:,iv) + reshape(klad, [2, n, 1]);
    end

    switch SMOOTHING
        case 'window'
            u_allp(j,:)=u(:);
            win=min(j-jw, is_window);
            u_av=1/win*reshape(sum(u_allp(j-win+1:j,:,:,:),1), [2, n, NT]);
        case 'spline'
            for a=1:2
                for b=1:n
                   u(a, b, :) = csaps(1:NT,u(a, b, :),spall(j),1:NT);
                end
            end
            u_allp(j,:)=u(:);
            win=min(j-jw, is_window);
            u_av=1/win*reshape(sum(u_allp(j-win+1:j,:,:,:),1), [2, n, NT]);
    end
    
    inner_toc=toc(inner_time);
    fprintf('loop time: %f \n', inner_toc);
    fprintf('remaining time(hr): %f \n', inner_toc*(nIS-j)/60/60);


end

outer_toc=toc(outer_time);

if clearup==1
    ss=ssp;
    J=Jp;
    F_all=F_allp;
    F_min_all=F_min_allp;
    u1_all=u_allp;
else
    ss=[ss ssp];
    J=[J Jp];
    F_all=[F_all F_allp];
    F_min_all=[F_min_all F_min_allp];
    u1_all=cat(1, u1_all, u_allp);
end

fprintf('simulation time (min): %f \n', outer_toc/60);
