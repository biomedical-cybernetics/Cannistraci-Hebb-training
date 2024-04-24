function [scores, CHA_info] = CHA_linkpred_bipartite(xb, is_path_based, methods, CHA_option, cores)

%%% INPUT %%%
% xb - Bipartite adjacency matrix of the network (unweighted).
%
% is_path_based - Use Path based computation or Node based computation.
%
% methods - Cell array of strings indicating the CH models to compute,
%   the possible options are: 'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'.
%   If empty or not given, methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'}.
%
% CHA_option - The possible options are:
%   CHA_option = 0 -> CHA is not computed.
%   CHA_option = 1 -> CHA is computed over the set of CH models indicated in "methods".
%   CHA_option = cell array of strings indicating a subset of the CH models in "methods" -> CHA is computed over this subset.
%   If empty or not given, CHA_option = 1 if length(methods)>1 and CHA_option = 0 if length(methods)==1.
%   The CHA computation is only valid over at least 2 methods.
%
% cores - number of cores to use for parallel computation.
%   Select 1 for serial computation.
%   If empty or not given, the maximum number available is used.
%
%
%%% OUTPUT %%%
% scores - Table containing CH-SPcorr scores for all node pairs.
%   The first two columns indicate the node pairs.
%   If CHA_option is not 0, the third column contains the scores of the CHA method.
%   Following columns contain the scores of each CH method with respective CH-SPcorr subranking.
%   Higher scores suggest higher likelihood of connection between the node pairs.
%
% CHA_info - Structure containing information about the CHA method in the fields:
%   methods -> cell array of strings indicating the CH models over which the CHA is computed
%   aupr -> for each CH model, aupr of discrimination between observed and non-observed links
%   selected_method -> string indicating the CH model selected by the CHA method
%   If CHA_option = 0, CHA_info = [].

% CHA method description:
% For each CH model, all the node pairs are assigned a rank-score (from 1 up)
% while ranking them by increasing CH scores and, in case of tie, by increasing CH-SPcorr scores.
% If they are still tied, they get the same rank-score.
% Therefore, the node pair with highest likelihood of connection gets the highest rank-score.
% For each CH model, using the rank-scores, the discrimination between observed and non-observed links is assessed by aupr.
% The CHA method selects the CH model with the highest aupr and provides in output its rank-scores.

% MEX support function:
% The Matlab function requires the MEX function "CH_scores_mex".
% Compile in Windows:
% Go to MATLAB "Add-Ons" and install "MATLAB Support for MinGW-w64 C/C++ Compiler"
% Build the MEX function using the following MATLAB command (change the MinGW path if needed):
% mex C:\ProgramData\MATLAB\SupportPackages\R2020b\3P.instrset\mingw_w64.instrset\lib\gcc\x86_64-w64-mingw32\6.3.0\libgomp.a CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% Compile in Linux or Apple Mac:
% Build the MEX functions using the following MATLAB commands:
% mex CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% It will generate a MEX file with platform-dependent extension,
% .mexw64 for Windows, .mexa64 for Linux, .mexmaci64 for Apple Mac.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check input
narginchk(1,6)
validateattributes(xb, {'numeric'}, {'binary'});
xb = sparse(xb);
if ~exist('methods', 'var')
    methods = [];
end
if ~exist('CHA_option', 'var')
    CHA_option = [];
end
if ~exist('cores', 'var')
    cores = [];
end

% reshape the bipartite adjacency matrix into monopartite
n1 = size(xb,1);
n2 = size(xb,2);
x = zeros(n1+n2);
x(1:n1,n1+1:end) = xb;
x(n1+1:end,1:n1) = xb';
x = sparse(x);

% compute monopartite link prediction scores
[scores, CHA_info] = CHA_linkpred_monopartite(x, is_path_based, methods, CHA_option, cores);

% exctract subset of bipartite scores
scores = scores(1:n1,n1+1:end);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [scores, CHA_info] = CHA_linkpred_monopartite(x, is_path_based, methods, CHA_option, cores)

%%% INPUT %%%
% x - monopartite adjacency matrix of the network (unweighted, undirected and zero-diagonal).
%
% is_path_based - Use Path based computation or Node based computation
%
% methods - Cell array of strings indicating the CH models to compute,
%   the possible options are: 'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'.
%   If empty or not given, methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'}.
%
% CHA_option - The possible options are:
%   CHA_option = 0 -> CHA is not computed.
%   CHA_option = 1 -> CHA is computed over the set of CH models indicated in "methods".
%   CHA_option = cell array of strings indicating a subset of the CH models in "methods" -> CHA is computed over this subset.
%   If empty or not given, CHA_option = 1 if length(methods)>1 and CHA_option = 0 if length(methods)==1.
%   The CHA computation is only valid over at least 2 methods.
%
% cores - number of cores to use for parallel computation.
%   Select 1 for serial computation.
%   If empty or not given, the maximum number available is used.
%
%
%%% OUTPUT %%%
% scores - Table containing CH-SPcorr scores for all node pairs.
%   The first two columns indicate the node pairs.
%   If CHA_option is not 0, the third column contains the scores of the CHA method.
%   Following columns contain the scores of each CH method with respective CH-SPcorr subranking.
%   Higher scores suggest higher likelihood of connection between the node pairs.
%
% CHA_info - Structure containing information about the CHA method in the fields:
%   methods -> cell array of strings indicating the CH models over which the CHA is computed
%   aupr -> for each CH model, aupr of discrimination between observed and non-observed links
%   selected_method -> string indicating the CH model selected by the CHA method
%   If CHA_option = 0, CHA_info = [].

% CHA method description:
% For each CH model, all the node pairs are assigned a rank-score (from 1 up)
% while ranking them by increasing CH scores and, in case of tie, by increasing CH-SPcorr scores.
% If they are still tied, they get the same rank-score.
% Therefore, the node pair with highest likelihood of connection gets the highest rank-score.
% For each CH model, using the rank-scores, the discrimination between observed and non-observed links is assessed by aupr.
% The CHA method selects the CH model with the highest aupr and provides in output its rank-scores.

% MEX support function:
% The Matlab function requires the MEX function "CH_scores_mex".
% Compile in Windows:
% Go to MATLAB "Add-Ons" and install "MATLAB Support for MinGW-w64 C/C++ Compiler"
% Build the MEX function using the following MATLAB command (change the MinGW path if needed):
% mex C:\ProgramData\MATLAB\SupportPackages\R2020b\3P.instrset\mingw_w64.instrset\lib\gcc\x86_64-w64-mingw32\6.3.0\libgomp.a CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% Compile in Linux or Apple Mac:
% Build the MEX functions using the following MATLAB commands:
% mex CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% It will generate a MEX file with platform-dependent extension,
% .mexw64 for Windows, .mexa64 for Linux, .mexmaci64 for Apple Mac.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check input
narginchk(1,6)
validateattributes(x, {'numeric'}, {'square','binary'});
x = sparse(x);
if ~issymmetric(x)
    error('The input matrix must be symmetric.')
end
if any(x(speye(size(x))==1))
    error('The input matrix must be zero-diagonal.')
end
if ~exist('methods', 'var') || isempty(methods)
    methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'};
else
    validateattributes(methods, {'cell'}, {});
    if any(~ismember(methods, {'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'}))
        error('Possible methods: ''RA_L2'',''CH1_L2'',''CH2_L2'',''CH3_L2'',''RA_L3'',''CH1_L3'',''CH2_L3'',''CH3_L3''.');
    end
    if length(methods) > length(unique(methods))
        error('The variable ''methods'' should not contain duplicates.')
    end
end
if ~exist('CHA_option', 'var') || isempty(CHA_option)
    if length(methods)==1
        CHA_option = [];
    else
        CHA_option = methods;
    end
elseif isnumeric(CHA_option)
    validateattributes(CHA_option, {'numeric'}, {'scalar','binary'});
    if CHA_option == 1
        CHA_option = methods;
    else
        CHA_option = [];
    end
else
    validateattributes(CHA_option, {'cell'}, {});
    if any(~ismember(CHA_option, methods))
        error('The variable ''CHA_option'' contains methods not present in ''methods''.');
    end
    if length(CHA_option) > length(unique(CHA_option))
        error('The variable ''CHA_option'' should not contain duplicates.')
    end
end
if length(CHA_option)==1
    error('The CHA computation is only valid over at least 2 methods.')
end
if ~exist('cores', 'var') || isempty(cores)
    cores = Inf;
else
    validateattributes(cores, {'numeric'}, {'scalar','integer','positive'});
end

% compute CH scores
M = length(methods);
L = NaN(M,1);
models = cell(M,1);
for m = 1:M
    temp = strsplit(methods{m},'_L');
    L(m) = str2double(temp{2});
    models{m} = temp{1};
end
L = unique(L);
models_all = {'RA','CH1','CH2','CH3'};
models = find(ismember(models_all,models))-1;
if isinf(cores)
    if is_path_based
        % Use path based method
        scores = CH_scores_mex(x, L, models);
    else
        % Use node based method
        scores = CH_scores_node_mex(x, L, models);
    end
else
    if is_path_based
        % Use path based method
        scores = CH_scores_mex(x, L, models, cores);
    else
        % Use node based method
        scores = CH_scores_node_mex(x, L, models, cores);
    end
end

scores = scores{1, 1};
CHA_info = [];
end