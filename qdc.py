import attr
import jax.numpy as jnp


@attr
class QDC:
    # attrs

    # methods
    def solve(self):
        U = dt*jnp.sum(u**2)
        Su=0.5*R * U
        S_girsanov = R * jnp.einsum(f'{A}t, j{A}t -> j', u, dW) #TODO: define A
        F = jnp.einsum('jμ, μν, jν -> j', jnp.conj(ψ), Qm, ψ) #TODO: define Qm
        S = -0.5* F + Su + S_girsanov
    
        F_all(p)=F;
    F_min_allp(j) =-2/Q*max(Send);

    fprintf('F: %f %f \n',F_allp(j),4/Q^2*sqrt(std(Send)))
    S=S+Send;
    %S=S+S_path+Send;
    
    Smin=min(S);
	S=S-Smin;
	w=exp(-S/lambda);

	Jp(j)=Smin-lambda*log(mean(w));
	w=w/sum(w);
	ssp(j)=1/sum(w.^2)/n_traj;

    if mod(j, nIS)==0
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
        plot(1:nIS,F_allp,1:nIS,F_min_allp)
        ylabel('F')
        drawnow

        figure(2)
        for a=1:n
        subplot(n,1,a)
        plot(dt*(1:NT),reshape(u_av(:,a,:), [2, NT])')
        end

%         figure(3)
%         subplot(3,1,1)
%         plot(dt*(1:NT),squeeze(m_all(:,:,1)),T,mphi(1),'k*');
%         axis([0 1.1*T -1 1]);
%         subplot(3,1,2)
%         plot(dt*(1:NT),squeeze(m_all(:,:,2)),T,mphi(2),'k*');
%         axis([0 1.1*T -1 1]);
%         subplot(3,1,3)
%         plot(dt*(1:NT),squeeze(m_all(:,:,3)),T,mphi(3),'k*');
%         axis([0 1.1*T -1 1]);

        figure(4)
        plot(dt*(1:NT),fid);
        axis([0 1.1*T 0 1]);

        [spec, freq] = getSpectrum(mean(fid, 1), T);
        figure(5)
        plot(freq, spec)

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
    u1_allp(j,:)=u(:);

    win=min(j-jw, is_window);
    u_av=1/win*reshape(sum(u1_allp(j-win+1:j,:,:,:),1), [2, n, NT]);