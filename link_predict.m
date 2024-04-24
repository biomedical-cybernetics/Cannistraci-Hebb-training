function score = link_predict(mask)

% input args:
%           - regrow_algorithm: the name of link predictor. Included:
%           'CHA', 'CN_L3', and 'SPM'.
%           - path: the path we save the adjacency matrix of each epoch
%           - n_connections: the number of links that we need to remove and
%           regrow.
%           - Norm: cumulative average among epochs or not. [0 or 1]
%           - remove_mode: remove also with link predictor or not. [0 or 1]
%           - layer: the exact layer of MLPs. [0, 1, 2]
%           - epoch: the exact epoch---use for cumulative
%           - concurrent: do remove and regrow concurrently or sequential
% returns:
%           - if remove_mode = 0, return only the new grown links.
%           - if remove_mode = 1, return both the removed links and new
%           grown links.


% read adjacency matrix and get scores of each link predictor
[score, CHA_info] = CHA_linkpred_bipartite_nosub(mask, 1, {'CH3_L3'}, 0);
score = score .* (mask == 0);
end


