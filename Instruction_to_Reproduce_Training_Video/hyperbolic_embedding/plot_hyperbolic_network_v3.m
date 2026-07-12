function plot_hyperbolic_network_v3(x, coords_native, representation_type, node_colors, name, node_shape)

%%% INPUT %%%
% x - adjacency matrix (NxN) of the network
%
% coords_native - polar [theta,radius] hyperbolic coordinates (Nx2) of the nodes in the native space
%
% representation_type - two options:
%   - 'poincare_unit_disk'
%   - 'native' (the radial coordinate is equal to the hyperbolic distance of the nodes to the origin of the Poincare unit disk)
%
% node_colors - three options:
%   - matrix (Nx3) indicating RGB colors for the nodes
%   - vector indicating numerical labels for the nodes (assumed consecutive integers starting from 1), colors will be assigned accordingly
%   - empty, in which case default colors will be assigned

% set colors
if nargin==3 || isempty(node_colors)
    hsv_colors = hsv(round(2*pi*1000));
    node_colors = hsv_colors(mod(round(coords_native(:,1)*1000),round(2*pi*1000))+1,:);
elseif isvector(node_colors)
    hsv_colors = hsv(max(node_colors));
    node_colors = hsv_colors(node_colors,:);
end

% coordinates transformation
coords_unit = coords_native;
coords_unit(:,2) = tanh(coords_native(:,2)/2);
[cart_coords_native(:,1),cart_coords_native(:,2)] = pol2cart(coords_native(:,1),coords_native(:,2));
[cart_coords_unit(:,1),cart_coords_unit(:,2)] = pol2cart(coords_unit(:,1),coords_unit(:,2));


%% native
if strcmp(representation_type, 'native')
    % plot circle
    hold on
    radius = max(coords_native(:,2));
    viscircles([0 0], radius, 'EdgeColor',[0.8 0.8 0.8]);
    
    % plot links
    [e1,e2] = find(triu(x,1));
    for i = 1:length(e1)
        if cart_coords_unit(e1(i),1) - cart_coords_unit(e2(i),1) + cart_coords_unit(e1(i),2) - cart_coords_unit(e2(i),2) ~= 0
        [~,~,cart_arc_1,cart_arc_2] = hyperbolic_line_v2([cart_coords_unit(e1(i),1),cart_coords_unit(e1(i),2)], [cart_coords_unit(e2(i),1),cart_coords_unit(e2(i),2)], 1, [0,0], 'step', 0.01, 0);
        pol_arc = NaN(length(cart_arc_1),2);
        [pol_arc(:,1),pol_arc(:,2)] = cart2pol(cart_arc_1,cart_arc_2);
        pol_arc(:,2) = 2*atanh(pol_arc(:,2));
        [cart_arc_1,cart_arc_2] = pol2cart(pol_arc(:,1),pol_arc(:,2));
        plot(cart_arc_1, cart_arc_2, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.7)
        end
    end
   
    %% plot nodes

if nargin>3 && nargin<6 
    nodesize = log(2+sum(x,2))*50;
   scatter(cart_coords_native(:,1),cart_coords_native(:,2),nodesize,node_colors,'filled','MarkerEdgeColor',[0.8 0.8 0.8]);
end
   
if nargin==6
    nodesize = log10(2+sum(x,2))*15;
    for i = 1:length(cart_coords_native(:,1))
     plot(cart_coords_native(i,1),cart_coords_native(i,2),node_shape{i},'markersize',nodesize(i),'markerfacecolor',node_colors(i,:),'MarkerEdgeColor',[0.8 0.8 0.8])   
    end
end

if nargin>4 && length(name) == length(cart_coords_native(:,1))
    for i = 1:length(cart_coords_native(:,1))
     text(cart_coords_native(i,1),cart_coords_native(i,2),[' ', name(i)])
    end
end

%% poincare

elseif strcmp(representation_type, 'poincare_unit_disk')
    % plot circle
    hold on
    radius = 1;
    viscircles([0 0], radius, 'EdgeColor', [0.8 0.8 0.8]);
    
    % plot links
    [e1,e2] = find(triu(x,1));
    for i = 1:length(e1)
        if cart_coords_unit(e1(i),1) - cart_coords_unit(e2(i),1) + cart_coords_unit(e1(i),2) - cart_coords_unit(e2(i),2) ~= 0
        [~,~,cart_arc_1,cart_arc_2] = hyperbolic_line_v2([cart_coords_unit(e1(i),1),cart_coords_unit(e1(i),2)], [cart_coords_unit(e2(i),1),cart_coords_unit(e2(i),2)], 1, [0,0], 'step', 0.01, 0);
        plot(cart_arc_1, cart_arc_2, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.7)
        end
    end
    
    % plot nodes
    if nargin>3 && nargin<6 
    nodesize = log(2+sum(x,2))*50;
    scatter(cart_coords_unit(:,1),cart_coords_unit(:,2),nodesize,node_colors,'filled','MarkerEdgeColor','k');
    end

if nargin==6
    nodesize = log10(2+sum(x,2))*15;
    for i = 1:length(cart_coords_unit(:,1))
     plot(cart_coords_unit(i,1),cart_coords_unit(i,2),node_shape{i},'markersize',nodesize(i),'markerfacecolor',node_colors(i,:),'MarkerEdgeColor',[0.8 0.8 0.8])   
    end
end


    if nargin>4 && length(name) == length(cart_coords_native(:,1))
    for i = 1:length(cart_coords_unit(:,1))
     text(cart_coords_unit(i,1),cart_coords_unit(i,2),[' ', name(i)])
    end
    end

end

% axis options
xlim([-radius, radius])
ylim([-radius, radius])
axis square
axis off
