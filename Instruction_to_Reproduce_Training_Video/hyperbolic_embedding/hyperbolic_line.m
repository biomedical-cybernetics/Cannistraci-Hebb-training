function [C_o, r_o, x_arc, y_arc] = hyperbolic_line(A, B, r, M, which, val, plotting)

%%% July 2017
%%% Josephine Thomas

%%%%%%%%%%%%%%%%%%%
%%% Description %%%
%%%%%%%%%%%%%%%%%%%

%%% Calculates the orthogonal circle (defined by center M_o and radius r_o) through two given euclidean
%%% points A,B within an original circle (defined by center M and radius r).
%%% Then calculates the euclidean coordinates x_arc, y_arc of points on the orthogonal
%%% circle beetween A and B (smaller arc) in euclidean coordinates.

%%% INPUT %%%
% M            - center of circle in euclidean coordinates in form [x,y]
% r            - radius of circle
% A,B          - two points within the circle in euclidean coordinates in form [x,y]
%                note: outside circle points could also work but give warning for imaginary parts
%                (to be checked for further improvements)
% which        - 'step' for giving stepsize in x_arc, y_arc
%                'n' for giving number of points in x_arc, y_arc
% val          - value for stepsize of arc or number of points on arc (depending on which)
% plotting     - '1' makes a plot of circle, orthogonal circle and arc

%%% OUTPUT %%%
% C_o          - center of orthogonal circle in euclidean coordinates in form [x,y]
% r_o          - radius of orthogonal circle
% x_arc, y_arc - x,y (euclidean) coordinates of the arc between points A and B

% NOTE: If A and B are on a line through the center, there is no need to
% find an orthogonal circle and M_o and r_o will be set to NaN.

%%%%%%%%%%%%%%%%%%%
%%% Input Check %%%
%%%%%%%%%%%%%%%%%%%

%%% Check if points A and B are lying within the disc %%%
if or(sqrt((M(1)-A(1))^2+(M(2)-A(2))^2)>r+10^(-15),sqrt((M(1)-B(1))^2+(M(2)-B(2))^2)>r+10^(-15)) %I use 10^(-15) because it seems matlab makes rounding errors in 10^(-16 or 17)
    warning('At least one of the points is lying outside the disc')
end

%%% Make sure that given points A and B are different %%%
if and(A(1) == B(1),A(2) == B(2))
    error('Please enter two different points')
end

%%%%%%%%%%%%%%%%%%%
%%% Calculation %%%
%%%%%%%%%%%%%%%%%%%

%%% In case points A and B are one a line through the center M of the original circle the hyperbolic line is just this straight line %%%

% Check if A and B are on a line through the center
if (abs(A(1) - B(1))) == 0  % are A and B on vertical line through center?
    tmp = (abs(A(1) - M(1))) < (10^-12);
elseif (abs(A(2)- B(2))) == 0 % are A and B on horizontal line through center?
    tmp = (abs(A(2))-abs(M(2))) < (10^-12);
else
    test = abs(M(2) - line_eq(A,B,M(1))); % catches all lines through center beside a vertical/horizontal ones
    tmp = test < 10^(-12);
end

% If A and B are on line through center make a straight line and set M_o and r_o to NaN
if tmp % (actually here just 2 points are needed for the line)
    if strcmp(which,'n')
        x_arc = linspace(A(1),B(1),val);
        y_arc = line_eq(A,B,x_arc);
    elseif strcmp(which,'step')
        x_arc = linspace(min(A(1),B(1)),max(B(1),A(1)),round((max(B(1),A(1))-min(A(1),B(1)))/0.01)+1);
        y_arc = line_eq(A,B,x_arc);
    else
        error('which has to be n or step')
    end
    C_o = [NaN,NaN];
    r_o = NaN;
    warning('Radius and center of orthogonal circle have been set to NaN, because A and B are on a line through the center.')
    
    % In all other cases make the hyperbolic line
else
    % Check if there will be a solution for the equations used below
    check = (A(1)*B(2)-B(1)*A(2));
    if check == 0
        error('no solution available')
    end
    % Calculate center M_o and radius r_o of the orthogonal circle
    [C_o,r_o] = ortho_circle(A, B, r, M);
    % Get cartesian coordinates of the arc between the points A and B on orthogonal circle
    [x_arc,y_arc] = get_arc(A, B, C_o, r_o, which, val);
end

%%%%%%%%%%%%%%%%
%%% Plotting %%%
%%%%%%%%%%%%%%%%

if plotting == 1 % plot circle, all the help lines and orthogonal circle and arc on the new circle
    
    figure();
    hold on
    
    % Coordinates of original circle
    phi = 0.01:0.01:2*pi;
    x = r.*cos(phi);
    y = r.*sin(phi);
    x = x + M(1);
    y = y + M(2);
    % Coordinates of orthogonal circle
    x_2 = r_o.*cos(phi); % make circle as if it were having its center in the origin
    y_2 = r_o.*sin(phi);
    x_2 = x_2 + C_o(1); % move center
    y_2 = y_2 + C_o(2);
    
    % Plot original circle
    plot(x,y,'b','MarkerSize',5)
    % Plot center of circle, M
    plot(M(1),M(2),'k.', 'Markersize',15)
    % Plot points A and B
    plot([A(1),B(1)],[A(2),B(2)],'g.','MarkerSize',15)
    % Plot orthogonal circle
    plot(x_2,y_2,'k','MarkerSize',5)
    % Plot arc between A and B
    plot(x_arc,y_arc,'-r','MarkerSize',10);
    
    axis equal
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Support Functions %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

function y = line_eq(point1, point2, x)
% Line equation from 2 points
% point1 and point2 need to be in form [x,y]

if (point2(1)-point1(1))==0
    error('vertical line')
else
    try
        y = calc_slope(point1,point2).*x + (point2(1)*point1(2)-point1(1)*point2(2))/(point2(1)-point1(1));
    catch
        y = repmat((point2(1)*point1(2)-point1(1)*point2(2))/(point2(1)-point1(1)),size(x));
    end
end

function m = calc_slope(point1, point2)
% Equation to calculate slope from 2 points
% point1 and point2 need to be in form [x,y]

dy = point2(2)-point1(2);
dx = point2(1)-point1(1);
if dx == 0
    error('vertical line, no proper slope defined')
elseif dy == 0
    error('horizontal line, slope 0')
else
    m = dy/dx;
end

function [Co,ro] = ortho_circle(A, B, r, C)
% Calculates the center Co and radius ro of the circle orthogonal to the
% circle defined by radius r and center C through the points A and B

%%% INPUT %%%
% r - radius of circle
% C - center of original circle in form [x y]
% A, B - points within the circle specified by center C and r

%%% OUTPUT %%%
% Co - center of the orthoganl circle in the form [x,y]
% ro - radius of orthogonal circle

% Distract origin from points A and B, so that the calculation as for a circle in
% the origin is possible
A(1) = A(1) - C(1);
A(2) = A(2) - C(2);
B(1) = B(1) - C(1);
B(2) = B(2) - C(2);

% Calculate center Co of orthogonal circle
tmp_A = A(1)^2 + A(2)^2 + r^2;
tmp_B = B(1)^2 + B(2)^2 + r^2;
Co_y = (0.5 * tmp_B - B(1) * tmp_A / (2 * A(1))) / (B(2) - B(1) * A(2) / A(1));
Co_x = (tmp_A - 2 * A(2) * Co_y) / (2 * A(1));

% Calculate radius r of orthogonal circle
ro = sqrt(Co_x^2 + Co_y^2- r^2);
% Add origin to center of orthogonal circle to shift back the complete construction of circles
Co = [Co_x + C(1), Co_y + C(2) ];

function phi = quadrant_phi(Px, Py, Cx, Cy, r)
% Checks, in which Quadrant of a circle defined by center [Cx,Cy] and radius r,
% a point [Px,Py] given in euclidean coordinates lies and calculates its angle accordingly

%%% INPUT %%%
% Px, Py - x and y coordinates of point to check
% Mx, My - x and y coordinates of the center of the circle
% r - radius of the circle

%%% OUTPUT %%%
% phi - angle of the point

if Py>=Cy && Px>=Cx %(1st)
    phi = acos((Px-Cx)/r);
elseif Py>=Cy && Px<Cx %(2rd)
    phi = acos((Px-Cx)/r);
elseif Py<Cy && Px<=Cx %(3rd)
    phi = 2*pi-acos((Px-Cx)/r);
elseif Py<Cy && Px>Cx %(4rd)
    phi = 2*pi - acos((Px-Cx)/r);
end

function [x_arc,y_arc] = get_arc(A, B, C, r, which, val)

% Check if given points lie on circle
tmp1 = (A(1)-C(1))^2+(A(2)-C(2))^2 - r^2;
tmp2 = (B(1)-C(1))^2+(B(2)-C(2))^2 - r^2;

% if tmp1>1e-10
%     warning('Point A does not lie exactly on the circle. x^2-Y^2 = %d \n This might be because the center of the given circle is not exact if you are using this function with the hyperbolic lines function', tmp1)
% elseif tmp2>1e-10
%     warning('Point B does not lie exactly on the circle. x^2-Y^2 = %d \n This might be because the center of the given circle is not exact if you are using this function with the hyperbolic lines function', tmp2)
% end

% Calculate angles of the given points on given circle
phi_A = quadrant_phi(A(1),A(2),C(1),C(2),r);
phi_B = quadrant_phi(B(1),B(2),C(1),C(2),r);

% Calculate shorter arc between the points on circle
if abs(phi_A-phi_B) >= pi
    tmp1 = 2*pi - max(phi_A,phi_B)+ min(phi_A,phi_B);
    if strcmp(which,'step')
        tmp2 = linspace(max(phi_A,phi_B),max(phi_A,phi_B)+tmp1,round(tmp1/(val/r))+1);
    elseif strcmp(which,'n')
        tmp2 = linspace(max(phi_A,phi_B),max(phi_A,phi_B)+tmp1,val);
    end
    phi_AB = mod(tmp2,2*pi);
else
    if strcmp(which,'step')
        phi_AB = linspace(min(phi_A,phi_B),max(phi_A,phi_B),round((max(phi_A,phi_B)-min(phi_A,phi_B))/(val/r))+1);
    elseif strcmp(which,'n')
        phi_AB = linspace(min(phi_A,phi_B),max(phi_A,phi_B),val);
    end
end

% Turn to cartesian coordinates
x_arc = r.*cos(phi_AB); % make circle as if it were having its center in the origin
y_arc = r.*sin(phi_AB);
x_arc = x_arc + C(1); % move center
y_arc = y_arc + C(2);
