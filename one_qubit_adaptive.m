%1 general program for learning time independent control edited on may 2015 for publ/jsp_special_issue
%clear all

rng('default');
% pauli matrices

sigmax=[0 1; 1 0];
sigmay=[0 -1i;1i 0];
sigmaz=[1 0;0 -1];

SMOOTHING = 'window';
%SMOOTHING = 'spline';
p=1e-4;    % cubic spline smoothing with 0 <= p <= 1. p=0 is straight line, p=1 is original data


T=1;                        % horizon time
D=0.01;					% diffusion constant in the Lindblad equation
R=0.01;						% diagonal u^2 cost R sum_i u_i^2
Q=10;						% size of quadratic end or path cost this is Q
n_traj=1000;
n_ref=1000;		% number of important sampling steps
NT=100;
dt=T/NT;
psi0=[1,1].';    % initial state
psi0=psi0/norm(psi0);
%psi0=[1,0]';

K=2;
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


Dtilde=D/2;                 % unraveling diffusion
lambda=Dtilde*R;

H0 = 0;

%pulses=u_av;

% end definitions -------------------------------------------------------

%load('~/projects/research/code/quantum control/GRAPE/grape_filtered_sols.mat');
%u_av = load('~/projects/research/code/quantum control/matlab/qcontrol/variations/one qubit/single_qubit_128K.mat').u_av;

u=zeros(2,NT);			    % all controls u(1:T) 
ss=zeros(1,n_ref);			% stores effective system size of local set of trajectories
J=zeros(1,n_ref);			% optimal cost-to-go per iteration
C=zeros(1,n_ref);			% optimal cost-to-go per iteration
m_all=zeros(NT,n_traj,3);    % stores single time dependent m trajectory
F_all=zeros(1,n_ref);
F_min_all=zeros(1,n_ref);
u_all=zeros(n_ref,2,NT);
u_av=zeros(2,NT);
%u_av = u_av + 0.5 * randn(2, NT);
%u_av = squeeze(us_grape(1, :, :));
u_norm=zeros(1,n_ref);
%pulses = squeeze(us(1, :, :));
%pulses=1 * randi([-10, 10], [2, K]);
%u_av=6*ones(2, NT);
% for k=1:K
%     iv=((k-1)*N2+1):(k*N2);
%     u_av(1,iv)=pulses(1, k);
%     u_av(2,iv)=pulses(2, k);
% end

%u_av = simple_resample(u, NT);


% chunk = floor(NT/32);
% for k=1:32
%     iv=((k-1)*chunk+1):(k*chunk);
%     u_av(1,iv)=pulses(1, k);
%     u_av(2,iv)=pulses(2, k);
% end

is_window=1;
jw= 0;
j0 = n_ref;

for j=1:n_ref
    tic
%    u_all(j,:,:) = u;
    if j > j0
        jw=j0;
        is_window=20;
    end
    u=u_av;
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

    % m dynamics
%     for t=1:NT
%         m(1,:)=m(1,:)+2*sqrt(Tf)*m(3,:).*(u(2,t)*dt+dWy(:,t)')-2*Dtilde*Tf*m(1,:)*dt;
%         m(2,:)=m(2,:)-2*sqrt(Tf)*m(3,:).*(u(1,t)*dt+dWx(:,t)')-2*Dtilde*Tf*m(2,:)*dt;
%         m(3,:)=m(3,:)+2*sqrt(Tf)*m(2,:).*(u(1,t)*dt+dWx(:,t)')-2*sqrt(Tf)*m(1,:).*(u(2,t)*dt+dWy(:,t)')-4*Dtilde*Tf*m(3,:)*dt;
%         m_all(t,:,:)=m';
%     end
%     norm_m = m(1,:).^2 + m(2,:).^2 + m(3,:).^2;
%     m = m./sqrt(norm_m);
    
    Su=0.5*R*dt*sum(sum(u.^2));
    u_norm(j)=Su;
    S = Su + R*sum(dWx.*(ones(n_traj,1)*u(1,:)),2)+R*sum(dWy.*(ones(n_traj,1)*u(2,:)),2);
    %fprintf('S: %f %f \n',mean(S),sqrt(std(S)))
    %S=S+R*(sum(dWx.^2,2)+sum(dWy.^2,2));
    %Send= -Q/2*(abs(psi'*phi)).^2 -Q/2*(abs(psi'*phi2)).^2; 
    Send= -Q/2*(abs(psi'*phi)).^2; 
    F_all(j)=-2/Q*mean(Send);
    F_min_all(j) =-2/Q*max(Send);
    %fprintf('F: %f %f \n',F_all(j),4/Q^2*sqrt(std(Send)));
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
    plot(1:n_ref,F_all,1:n_ref,F_min_all)
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