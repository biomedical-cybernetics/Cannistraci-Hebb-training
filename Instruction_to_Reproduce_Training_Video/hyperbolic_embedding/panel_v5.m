
function panel_v5


% load data_noin_v6.mat % noin means no input layer (no retina) 
load active_neurons.mat

% pw={'EBC','RA1','RA2'}; % preweighting method
% ndr={'ncISO','ISO','ncMCE','MCE','LE'}; % nonlinear dimension reduction method
% aa={'EA','original'};
% cas=1;
% repr= 'native'; % 'poincare_unit_disk'

pw={'EBC','RA1','RA2'}; % preweighting method
ndr={'ncISO','ISO','ncMCE','MCE','LE'}; % nonlinear dimension reduction method
aa={'EA'}; %,'original'};
cas=1;
repr= 'native';  % 'poincare_unit_disk'

 
for i=1:length(pw)
    for j=1:length(ndr)
        for k=1:length(aa)

figure
coords = coalescent_embedding_v1(x_symetric,pw{i},ndr{j},aa{k}, 2);

%plot_hyperbolic_network_v3(x_symetric, coords,repr,communitylayer1, labels1,communitysymbol) 
plot_hyperbolic_network_v3(x_symetric, coords,repr,communitycolor, [], communitysymbol) % Eunhye for article: communityshape->communitylayer3 []->labels1
if cas == 1
[index, g, pvalue] = compute_angular_separation(coords(:,1), communitylayer, 0);
title([pw{i},'-',ndr{j},'-',aa{k},', ','ASI=',num2str(index),', ','p-value=',num2str(pvalue)])
else
title([pw{i},'-',ndr{j},'-',aa{k}])
end

        end
    end
end
