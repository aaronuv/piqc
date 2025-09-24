% Annealing PiQC for the single qubit

rng('default');
% pauli matrices

sigmax=[0 1; 1 0];
sigmay=[0 -1i;1i 0];
sigmaz=[1 0;0 -1];

SMOOTHING = 'window';
%SMOOTHING = 'spline';
p=1e-4;    % cubic spline smoothing with 0 <= p <= 1. p=0 is straight line, p=1 is original data

Dfinal=1e-10;
Dinit=1e-1;


%DSCHEDULE='linear';
%DSCHEDULE='1/j';
%DSCHEDULE='1/j^2';
%DSCHEDULE='exp';
%DSCHEDULE='const + exp';
DSCHEDULE='steps';
%DSCHEDULE='logsteps';
%DSCHEDULE='ct';


T=1;                        % horizon time
D=0.001;					% diffusion constant in the Lindblad equation
R=1;						% diagonal u^2 cost R sum_i u_i^2
Q=100;						% size of quadratic end or path cost this is Q
n_traj=400;
nIS=500;		% number of important sampling steps
NT=100;
dt=T/NT;
psi0=[1,1].';    % initial state
psi0=psi0/norm(psi0);

K=50;
N2=NT/K;

%phi=1/sqrt(2)*[1,1]';    % target state
phi=[1,1i].';
phi=phi/norm(phi);


m0(1)=psi0'*sigmax*psi0;
m0(2)=psi0'*sigmay*psi0;
m0(3)=psi0'*sigmaz*psi0;

mphi(1)=real(phi'*sigmax*phi);
mphi(2)=real(phi'*sigmay*phi);
mphi(3)=real(phi'*sigmaz*phi);


H = rand(2, 2) + 1i * rand(2, 2);
H = H + H';

energy_ground_truth = min(eig(full(H), 'vector'));

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
    nsteps=20;
    lengthstep=nIS/nsteps;
    for k=1:nsteps
        Dall((k-1)*lengthstep + (1:lengthstep))=Dinit*(Dfinal/Dinit)^(k/nsteps);
    end

end

% end definitions -------------------------------------------------------

u=zeros(2,NT);			    % all controls u(1:T) 
ss=zeros(1,nIS);			% stores effective system size of local set of trajectories
J=zeros(1,nIS);			% optimal cost-to-go per iteration
C=zeros(1,nIS);			% optimal cost-to-go per iteration
m_all=zeros(NT,n_traj,3);    % stores single time dependent m trajectory
F_all=zeros(1,nIS);
F_min_all=zeros(1,nIS);
u_all=zeros(nIS,2,NT);
u_av=zeros(2,NT);
u_norm=zeros(1,nIS);

H0 = 0;


is_window = 1;
jw = 0;
j0 = 20;

for j=1:nIS
    tic
    if j > j0
        jw=j0;
        is_window=10;
    end
    u=u_av;

    D=Dall(j);
    Dtilde=D/2;                 % unraveling diffusion

    lambda=Dtilde*R;

    % initialize 
    psi=psi0*ones(1,n_traj);
    m=m0'*ones(1,n_traj);
    % generate trajectories 
    dWx=sqrt(Dtilde*dt)*randn(n_traj,NT);	% stores all noise realizations
    dWy=sqrt(Dtilde*dt)*randn(n_traj,NT);	% stores all noise realizations

    % psi dynamics
    for t=1:NT
        psi=psi -1i*H0*psi*dt -Dtilde*psi*dt -1i*(sigmax*psi).*(ones(2,1)*(u(1,t)*dt+dWx(:,t)')) -1i*(sigmay*psi).*(ones(2,1)*(u(2,t)*dt+dWy(:,t)'));
        norm_psi = sum(psi.*conj(psi),1);
        psi=psi./sqrt(norm_psi);

        m(1,:)=sum(conj(psi).*(sigmax*psi),1);
        m(2,:)=sum(conj(psi).*(sigmay*psi),1);
        m(3,:)=sum(conj(psi).*(sigmaz*psi),1);
        m_all(t,:,:)=m';
    end

    Su=0.5*R*dt*sum(sum(u.^2));
    u_norm(j)=Su;
    S = Su + R*sum(dWx.*(ones(n_traj,1)*u(1,:)),2)+R*sum(dWy.*(ones(n_traj,1)*u(2,:)),2);
    F = (abs(psi'*phi)).^2;
    Send= -Q/2*F; 
    F_all(j)=mean(F);
    F_min_all(j) =min(F);
    S=S+Send;

    C(j)=mean(S);

    Smin=min(S);
    S=S-Smin;
    w=exp(-S/lambda);

    J(j)=Smin-lambda*log(mean(w));
    w=w/sum(w);
    ss(j)=1/sum(w.^2)/n_traj;

    if mod(j, 40)==0
        figure(1)
        subplot(3,2,1)
        plot(ss)
        ylabel('ss')
        subplot(3,2,2)
        plot(J, 'b')
        hold on
        plot(C, 'r')
        hold off
        ylabel('cost')
        subplot(3,2,3)
        plot(dt*(1:NT), u')
        ylabel('u')
        subplot(3,2,4)
        plot(u_norm)
        ylabel('norm u')
        subplot(3,2,5)
        plot(1:nIS,F_all,1:nIS,F_min_all)
        ylabel('F')
        drawnow

        figure(2)
        subplot(3,1,1)
        plot(dt*(1:NT),squeeze(m_all(:,:,1)),T,mphi(1),'k*');
        axis([0 1.1*T -1 1]);
        subplot(3,1,2)
        plot(dt*(1:NT),squeeze(m_all(:,:,2)),T,mphi(2),'k*');
        axis([0 1.1*T -1 1]);
        subplot(3,1,3)
        plot(dt*(1:NT),squeeze(m_all(:,:,3)),T,mphi(3),'k*');
        axis([0 1.1*T -1 1]);
        drawnow
    end

    % K equal time intervals
    % N is divisible by K
    for k=1:K
        iv=((k-1)*N2+1):(k*N2);
        klad1=1/dt/N2*sum(sum((w*ones(1,length(iv))).*dWx(:,iv)));
        klad2=1/dt/N2*sum(sum((w*ones(1,length(iv))).*dWy(:,iv)));
        u(1,iv)=u(1,iv) + klad1;
        u(2,iv)=u(2,iv) + klad2;
    end
    u_all(j,:,:)=u;
    win=min(j-jw,is_window);
    u_av=1/win*squeeze(sum(u_all(j-win+1:j,:,:),1));

    switch SMOOTHING
        case 'window'
            u_all(j,:,:) = u;
            win = min(j,is_window);
            u_av = 1/win*squeeze(sum(u_all(j-win+1:j,:,:),1));
        case 'spline'
            %u(1,:) = csaps(1:NT,u(1,:),p,1:NT, ww);
            %u(2,:) = csaps(1:NT,u(2,:),p,1:NT, ww);
            u(1,:) = csaps(1:NT,u(1,:),p,1:NT);
            u(2,:) = csaps(1:NT,u(2,:),p,1:NT);
            u_all(j,:,:) = u;
            win = min(j,is_window);
            u_av = 1/win*squeeze(sum(u_all(j-win+1:j,:,:),1));
    end

end

fprintf('F: %f \n',mean(F_all(end-20:end)));