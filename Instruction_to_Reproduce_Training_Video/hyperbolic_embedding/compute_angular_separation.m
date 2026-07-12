function [index, group_index, pvalue] = compute_angular_separation(coords, labels, show_plot, rand_reps, rand_seed, worst_comp)

% MATLAB implementation of the angular separation index (ASI):
% a quantitative measure to evaluate the separation of groups
% over the circle circumference (2D) or sphere surface (3D).

% Reference:
% A. Muscoloni and C. V. Cannistraci (2019), "Angular separability of data clusters or network communities
% in geometrical space and its relevance to hyperbolic embedding", arXiv:1907.00025

% Released under MIT License
% Copyright (c) 2019 A. Muscoloni, C. V. Cannistraci

%%% INPUT %%%
% coords - 2D case: Nx1 vector containing for each sample the angular coordinates
%          3D case: Nx2 matrix containing for each sample the coordinates (azimut,elevation)
%          angular and azimuth coordinates must be in [0,2pi]
%          elevation coordinates must be in [-pi/2,pi/2]
%          the code automatically selects the 2D or 3D case depending on the size of the "coords" variable
% labels - N labels for the samples indicating the group membership (numeric vector or cell of strings)
% show_plot - [optional] 1 or 0 to indicate whether the plot of the results has to be shown or not (default = 1)
% rand_reps - [optional] repetitions for evaluating random coordinates (default = 1000)
% rand_seed - [optional] nonnegative integer seed for random number generator (by default a seed is created based on the current time)
% worst_comp - [optional] 1 or 0 to indicate if the worst case should be approximated computationally or theoretically (default = 1)
%              note that for the 3D case only the value 1 is valid
% (NB: optional inputs not given or empty assume the default value)

%%% OUTPUT %%%
% index - overall index in [0,1], a value 1 indicates that all the groups
%         are perfectly separated over the circle circumference (2D) or sphere surface (3D),
%         the more the groups are mixed the more the index tends to 0, representing a worst-case scenario.
% group_index - vector containing an index in [0,1] for each group,
%               to assess its separation with respect to the other groups
% pvalue - empirical p-value computed comparing the observed index with a null distribution
%          of indexes obtained from random permutations of the coordinates

% check input
narginchk(2,6)
validateattributes(coords, {'numeric'}, {})
N = size(coords,1);
D = size(coords,2) + 1; 
if D == 2
    if any(coords(:,1)<0 | coords(:,1)>2*pi)
        error('Angular coordinates must be in [0,2pi]')
    end
    if N < 4
        error('The index in 2D cannot be assessed for less than 4 samples')
    end
elseif D == 3
    if any(coords(:,1)<0 | coords(:,1)>2*pi)
        error('Azimuth coordinates must be in [0,2pi]')
    end
    if any(coords(:,2)<-pi/2 | coords(:,2)>pi/2)
        error('Elevation coordinates must be in [-pi/2,pi/2]')
    end
    if N < 6
        error('The index in 3D cannot be assessed for less than 6 samples')
    end
else
    error('Input coordinates must be a one-column vector (2D case) or two-columns matrix (3D case)')
end
validateattributes(labels, {'numeric','cell'}, {'vector','numel',N})
if ~exist('show_plot','var') || isempty(show_plot)
    show_plot = 1;
else
    validateattributes(show_plot, {'numeric'}, {'scalar','binary'})
end
if ~exist('rand_reps','var') || isempty(rand_reps)
    rand_reps = 1000;
else
    validateattributes(rand_reps, {'numeric'}, {'scalar','integer','positive'})
end
if ~exist('rand_seed','var') || isempty(rand_seed)
    rand_str = RandStream('mt19937ar','Seed','shuffle');
else
    validateattributes(rand_seed, {'numeric'}, {'scalar','integer','nonnegative'})
    rand_str = RandStream('mt19937ar','Seed',rand_seed);
end
if ~exist('worst_comp','var') || isempty(worst_comp)
    worst_comp = 1;
else
    validateattributes(worst_comp, {'numeric'}, {'scalar','binary'})
    if D == 3 && worst_comp == 0
        error('In 3D the worst case can be approximated only computationally (worst_comp = 1)')
    end
end

% convert labels
unique_labels = unique(labels);
M = length(unique(labels));
if M==1 || M==N
    error('The number of groups must be greater than 1 and lower than the number of samples')
end
temp = zeros(N,1);
Nk = zeros(M,1);
for k = 1:M
    if isnumeric(labels)
        temp(labels==unique_labels(k)) = k;
    else
        temp(strcmp(labels,unique_labels{k})) = k;
    end
    Nk(k) = sum(temp == k);
end
labels = temp; clear temp;

% compute index
[index, group_index, pvalue, index_rand] = compute_index(D, coords, labels, N, Nk, M, rand_reps, rand_str, worst_comp);

% restore original labels in group index
if isnumeric(unique_labels)
    group_index(:,1) = unique_labels;
else
    group_index = num2cell(group_index);
    group_index(:,1) = unique_labels(:);
end

% plot results
if show_plot
    plot_results(index, index_rand, pvalue)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [index, group_index, pvalue, index_rand] = compute_index(D, coords, labels, N, Nk, M, rand_reps, rand_str, worst_comp)

if D == 2
    compute_mistakes = @compute_mistakes_2D;
else
    compute_mistakes = @compute_mistakes_3D;
end

% compute mistakes in input coordinates
mistakes = compute_mistakes(coords, labels, N, Nk, M);

% compute mistakes in random coordinates
mistakes_rand = zeros(M,rand_reps);
for i = 1:rand_reps
    idx_rand = randperm(rand_str, size(coords,1));
    mistakes_rand(:,i) = compute_mistakes(coords(idx_rand,:), labels, N, Nk, M);
end

if all(isnan(mistakes)) || all(isnan(mistakes_rand(:)))
    error('The index could not be computed for any group.')
end

% find the worst case
if worst_comp == 1
    [~,idx] = max(nansum(mistakes_rand,1));
    mistakes_worst = mistakes_rand(:,idx);
else
    mistakes_worst = ceil((N-Nk).*(Nk-1)./Nk);
    mistakes_worst(Nk == 1) = NaN;
end

% compute group index
group_index = zeros(M,2);
group_index(:,2) = 1 - mistakes./mistakes_worst;

% compute overall index
index = 1 - nansum(mistakes)/nansum(mistakes_worst);
index = max(index,0);

% compute pvalue
index_rand = 1 - nansum(mistakes_rand,1)./repmat(nansum(mistakes_worst),1,rand_reps);
index_rand = max(index_rand,0);
pvalue = (sum(index_rand >= index) + 1) / (rand_reps + 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_results(index, index_rand, pvalue)

% plot figure
figure('color', 'white')
[fy,fx] = ksdensity(index_rand);
plot(fx, fy, 'k', 'LineWidth', 2)
hold on
plot([index,index], [0,max(fy)*1.1], 'r', 'LineWidth', 2)
set(gca,'YLim',[0,max(fy)*1.1],'XTick',0:0.1:1,'XLim',[0,max(max(fx),index)*1.1])
box on
xlabel('index')
ylabel('probability density')
text(1, 1.05, ['pvalue = ' num2str(pvalue)], 'Units', 'normalized', 'HorizontalAlignment', 'right')
legend({'null distribution','observed value'},'Location','northoutside','Orientation','vertical')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% 2D separation (circle circumference) %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mistakes = compute_mistakes_2D(coords, labels, N, Nk, M)

% ranking of the samples
x = zeros(N,1);
[~,idx] = sort(coords);
x(idx) = 1:N;

% for each group
mistakes = zeros(M,1);
for k = 1:M

    if Nk(k) == 1
        mistakes(k) = NaN;
        continue;
    end
    
    % compute number of wrong samples within the extremes of the group,
    % where the extremes are the adjacent samples at the maximum distance:
    % - find the number of samples of other groups between adjacent samples
    %   of the current group
    % - sum them excluding the maximum value, since it will be related to
    %   wrong samples outside the extremes

    x_k = x(labels == k);
    x_list = sort(x_k);
    wr = zeros(Nk(k),1);
    for l = 1:Nk(k)-1
        wr(l) = x_list(l+1)-x_list(l)-1;
    end
    wr(end) = N-x_list(end)+x_list(1)-1;
    [~,max_id] = max(wr);
    wr(max_id) = [];
    mistakes(k) = sum(wr);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% 3D separation (sphere surface) %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mistakes = compute_mistakes_3D(coords, labels, N, Nk, M)

% rank samples for azimuth
azim_rank = zeros(N,1);
[~,idx] = sort(coords(:,1));
azim_rank(idx) = 1:N;

mistakes = zeros(M,1);
for k = 1:M
    
    if Nk(k) < 3
        mistakes(k) = NaN;
        continue;
    end
    
    try
        % map the samples between the group extremes to a rectangular 2D area
        [xy_group, xy_other] = map_samples_between_group_extremes_3D(labels==k, azim_rank, coords(:,1), coords(:,2), N, Nk(k));
        
        if isempty(xy_other)
            mistakes(k) = 0;
        else
            % compute mistakes within the polygonal area delimited by the group samples
            pol_idx = convhull(xy_group(:,1),xy_group(:,2));
            [in_pol, on_pol] = inpolygon(xy_other(:,1),xy_other(:,2),xy_group(pol_idx,1),xy_group(pol_idx,2));
            mistakes(k) = sum(in_pol) - sum(on_pol);
        end
    catch
        mistakes(k) = NaN;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xy_group, xy_other] = map_samples_between_group_extremes_3D(labels_k, azim_rank, azim, elev, N, Nk)

% find group extremes: azimuth
azim_k_order = sort(azim_rank(labels_k));
wr = zeros(Nk,1);
for i = 1:Nk-1
    wr(i) = azim_k_order(i+1) - azim_k_order(i) - 1;
end
wr(Nk) = N - azim_k_order(end) + azim_k_order(1) - 1;
[~,max_idx] = max(wr);
if max_idx == Nk
    azim_ext = [azim(azim_rank==azim_k_order(1)) azim(azim_rank==azim_k_order(Nk))];
    disc = 0;
else
    azim_ext = [azim(azim_rank==azim_k_order(max_idx)) azim(azim_rank==azim_k_order(max_idx+1))];
    disc = 1;
end

% find group extremes: elevation
elev_ext = [min(elev(labels_k)) max(elev(labels_k))];

% detect samples between the group extremes
detected = (elev>=elev_ext(1) & elev<=elev_ext(2)) & ...
    ((~disc & (azim>=azim_ext(1) & azim<=azim_ext(2))) | (disc & (azim>=azim_ext(2) | azim<=azim_ext(1))));

% map the coordinates to a rectangular 2D area
if ~disc
    x_detected = azim(detected) - azim_ext(1);
else
    x_detected = mod(azim(detected) + (2*pi-azim_ext(2)), 2*pi);
end
y_detected = elev(detected) - elev_ext(1);

% divide samples belonging to the group and not
xy_group = [x_detected(labels_k(detected)==1) y_detected(labels_k(detected)==1)];
xy_other = [x_detected(labels_k(detected)==0) y_detected(labels_k(detected)==0)];
