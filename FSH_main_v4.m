function Output_matrix=FSH_main_v3(f_mat, s_mat)
% Original Code is for Functional-by-Structural Hierarchical Mapping (FSH mapping)
% author: Liang Zhan
% Date: March-04-2017
% If any problem, please contact:  zhan.liang@gmail.com

% input: two N x N x M matrices,  N is the number of nodes and M is the number of subjects
% 		 f_mat is functional network (derived from functional MRI), 
% 		 s_mat is structural network (derived from diffusion MRI), 

% output: Output_matrix is a binary usage matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%% Reference:  
% 1) Leow, A.D., Zhan, L., Arienzo, D., GadElkarim, J.J., Zhang, A.F., Ajilore, O., Kumar, A., Thompson, P.M. and Feusner, J.D., (2012), Hierarchical structural mapping for globally optimized estimation of functional networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 228-236). Springer, Berlin, Heidelberg.
% 2) Ajilore, O., Zhan, L., GadElkarim, J., Zhang, A., Feusner, J., Yang, S., Thompson, P.M., Kumar, A. and Leow, A., (2013). Constructing the resting state structural connectome. Frontiers in neuroinformatics, 7, p.30.
% 3) Korthauer, L.E., Zhan, L., Ajilore, O., Leow, A. and Driscoll, I., (2018). Disrupted topology of the resting state structural connectome in middle-aged APOE ε4 carriers. NeuroImage.


nNodes=size(f_mat,1);
mask=1-diag(ones(nNodes,1));

for i=1:size(f_mat,3)
	f_mat(:,:,i)=f_mat(:,:,i).*mask+(1-mask); 
end


 % this is to remove diagonal elements 
for i=1:size(s_mat,3)
    s_mat(:,:,i)=s_mat(:,:,i).*mask;   
end

seed=1; % this is a random seed, and should be assigned to a different value if you want to run multiple times.
Output_matrix=FSH_entrance(f_mat,s_mat,seed);   

end


function final_result=FSH_entrance(f_mat,s_mat,seed)
 nNodes=size(f_mat,1);
 mask=1-diag(ones(nNodes,1));
 SA=ones(nNodes).*mask;
 current=SA;
 itry=0;
 success=0;
 consec=0;
 consec1=0;
 max_try=10000;
 max_success=nNodes^2;
 max_consec_rejections =2000;
 init_T=1;  % initial temperature
 stop_T=1e-6; % final temperature
 T=init_T;
 % compute initial energy
 initenergy = cal_whole_energy(SA,f_mat,s_mat,mask);
 old_energy = initenergy;
 
 % keep track of best solution
 absMax = old_energy;             absMaxConf = current;
 rng(seed);
 while true
      itry = itry+1;
      % % Stop / decrement T criteria
      if itry >= max_try || success >= max_success || consec1>= max_consec_rejections
        if T < stop_T || consec >= max_consec_rejections
            break;
        else
            T = cool_tmp(T);  % decrease T according to cooling schedule
            itry = 1;
            success = 0;
            consec1 = 0;
            consec = 0;
        end
      end
      
      % then the simulated annealing is to randomly swap aij and aji
	  px = randi(nNodes,2,1); 
      newparam1 = current;
      newparam1(px(1),px(2)) =1- current(px(1),px(2));
      newparam1(px(2),px(1)) =1- current(px(2),px(1));
      
      new_energy =cal_whole_energy(newparam1,f_mat,s_mat,mask);    
      if (old_energy>new_energy)
          current = newparam1;
          old_energy = new_energy;
          success = success+1;        
          consec = 0;
      else      
          if (exp((old_energy-new_energy)/T)>rand)
              current = newparam1;
              old_energy = new_energy;
              success = success+1;
          else
              consec = consec+1;
          end
      end
      if (new_energy<absMax)
        absMax=new_energy;
        absMaxConf=current;
      else
        consec1=consec1+1; 
      end
 end
 
 final_result=absMaxConf;
end

function total_energy=cal_whole_energy(SA,f_mat,s_mat,mask)
total_energy=0;
for i=1:size(f_mat,3)
    subj_f_mat=f_mat(:,:,i);
    subj_s_mat=s_mat(:,:,i);
    individual_energy=cal_individual_energy(SA,subj_f_mat,subj_s_mat,mask);
    total_energy=total_energy+individual_energy;
end
end

function individual_energy=cal_individual_energy(SA,funtional_mat,structural_mat,mask)
warning off
V=funtional_mat;
C=structural_mat;
tmp=1./(SA.*C); % for the purpose of input of Dijkstra's algorithm
tmpx=-distance_wei(tmp);
tmpx(isinf(tmpx))=-1000000;
tmpx=tmpx.*mask;

[~,r] = nlinfit(tmpx(:),V(:),@mymodel,1);

% individual_energy=sum(abs(r)); %Keegan suggested making this |r| or r^2 so that sign of residuals does not affect the sum
individual_energy=r'*r; % this is to use r^2 as the residue

end


function F=mymodel(beta, x)
F=exp(beta*x);
F(isnan(F))=0;
F(isinf(F))=0;
end 

function D=distance_wei(G)
%DISTANCE_WEI       Distance matrix
%
%   D = distance_wei(W);
%
%   The distance matrix contains lengths of shortest paths between all
%   pairs of nodes. An entry (u,v) represents the length of shortest path 
%   from node u to node v. The average shortest path length is the 
%   characteristic path length of the network.
%
%   Input:      W,      weighted directed/undirected connection matrix
%
%   Output:     D,      distance matrix
%
%   Notes:
%       The input matrix must be a mapping from weight to distance. For 
%   instance, in a weighted correlation network, higher correlations are 
%   more naturally interpreted as shorter distances, and the input matrix 
%   should consequently be some inverse of the connectivity matrix.
%       Lengths between disconnected nodes are set to Inf.
%       Lengths on the main diagonal are set to 0.
%
%   Algorithm: Dijkstra's algorithm.
%
%
%   Mika Rubinov, UNSW, 2007-2010.

%Modification history
%2007: original
%2009-08-04: min() function vectorized

n=length(G);
D=zeros(n); D(~eye(n))=inf;                 %distance matrix

for u=1:n
    S=true(1,n);                            %distance permanence (true is temporary)
    G1=G;
    V=u;
    while 1
        S(V)=0;                             %distance u->V is now permanent
        G1(:,V)=0;                          %no in-edges as already shortest
        for v=V
            W=find(G1(v,:));                %neighbours of shortest nodes
            D(u,W)=min([D(u,W);D(u,v)+G1(v,W)]); %smallest of old/new path lengths
        end

        minD=min(D(u,S));
        if isempty(minD)||isinf(minD),      %isempty: all nodes reached;
            break,                          %isinf: some nodes cannot be reached
        end;

        V=find(D(u,:)==minD);
    end
end
end

function new_T=cool_tmp(old_T)
 new_T=old_T*0.75;
end
