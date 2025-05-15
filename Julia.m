clc 
clear
rng(2141444)
load('MPI_Julia_True.mat');
[xq,yq] = meshgrid(linspace(-1,1,400), linspace(-1,1,400));
xz=xq(:); yz=yq(:); xz=xz(ZZ==1); yz=yz(ZZ==1);
%% *************************** Dynamics ***********************************
state_dimension = 2;
julia_parameters = [-0.7 ; 0.2];
f =  @(t,x)([ x(1,:).^2 - x(2,:).^2 + julia_parameters(1) ;
           2*x(1,:).*x(2,:) + julia_parameters(2) ]); % Julia recursion
%% ************************** Initialization ********************************

% Setting parameters
N_epoch = 50;
  basis = 'rbf';        % rbf: Radial basis functions, monomial: monomial basis functions
 approx = 'least';      % least: Least squares, linrpog: Linear programing
    con = 'smooth';     % smooth: smooth function for constraint, indicator: indicator function for constraint
 method = 'grid';       % random: random data and basis centers, grid: grid data and basis centers
      d = 70;           % basis dimention, 20 for monomial is 
     Ns = 100;          % sampling volume per dimension
  Tresh = 1;          % Trashold for having some data ans basis outside the boundary 
      a = 0;          % Set 0 for non-adaptive treshold
      
% gif settings
figure(1)
filename = 'Julia.gif';  

% Collect data
tic
disp('Starting data collection')
switch method                                  
    case 'random'
        x = 1.1*rand_sphere(state_dimension,Ns^2);
        
    case 'grid'
        sampx = repmat(linspace(-1,1,Ns),1,Ns);
        sampy = repmat(linspace(-1,1,Ns),Ns,1); sampy=sampy(:)';
            x = [sampx; sampy];
end

% Constraint
switch con                                  
    case 'smooth'
        g = x(1,:).^2 + x(2,:).^2;
    case 'indicator'
        g = 1.01*(x(1,:).^2 + x(2,:).^2>1);
end

% Post processing the data
x_indx= (g<=Tresh);
x = x(:,x_indx);                % Cutting unnecessary data to increase performance 
x_plus = f(0,x);

V = g(x_indx);                  % Maximum invariant set = constraint set (initialization)
fprintf('Data collection DONE, time = %1.2f s \n\n', toc);

%% ******************* Basis functions and lifting ************************
switch method                                  
    case 'random'
        cent = rand_sphere(state_dimension,d^2);
        
    case 'grid'
        x_cent = repmat(linspace(-1,1,d),1,d);
        y_cent = repmat(linspace(-1,1,d),d,1); y_cent=y_cent(:)';
        cent = [x_cent; y_cent];
end
g_cent = griddata(x(1,:), x(2,:), V, cent(1,:), cent(2,:));    % Interpolating g in basis centers to be cut
cent = cent(:,g_cent<=Tresh);                      % Cutting centers outside the invariant set to increase performance
n_basis = length(cent);             
basisEval = @(x)(rbf(x,cent,'thinplate') );   
liftFun = @(xx)(basisEval(xx));

disp('Starting LIFTING')
tic
phi = liftFun(x);
phi_plus = liftFun(x_plus);
fprintf('Lifting DONE, time = %1.2f s \n\n', toc);

%% ********************* Starting the algorithm ***************************
fprintf('You will have %d epoches. \n****Sit Tight and Good Luck!****\n\n',N_epoch)
for epoch=1:N_epoch
    fprintf('************* STARTING %dth EPOCH  *************\n', epoch);

% adjusting data for the next epoch    
if epoch<=N_epoch/2
     Tresh = 1+a*(N_epoch-2*epoch)/(N_epoch);                             % We can impliment adaptive Treshold
end
    x_indx = (V<=Tresh);                                         % Finding data indices inside the invariant set to pick

switch basis
    case 'rbf'
    h_cent = griddata(x(1,:), x(2,:), V, cent(1,:), cent(2,:));  % Interpolating the value of h at basis centers
 cent_indx = (h_cent<=Tresh);                                    % Finding center indices inside the invariant set to pick
      cent = cent(:,cent_indx);
       phi = phi(cent_indx,x_indx);                              % Picking both data and basis functions inside the invariant set
  phi_plus = phi_plus(cent_indx,x_indx); 

    case 'monomial'
       phi = phi(:,x_indx);                                       % Picking both data and basis functions inside the invariant set
  phi_plus = phi_plus(:,x_indx);
end
    V = V(x_indx);
    
disp('Starting APPROXIMATION')
    tic
switch approx                                  
    case 'least'
        theta = (V*phi')/(phi*phi');
    case 'linprog'
        theta = linprog(sum(phi,2)./length(x),-phi',-V)';
end 
Alift = (phi_plus*phi')/(phi*phi');
    V = max(theta*phi, theta*Alift*phi);          % Updating h
fprintf('Approximation DONE, time = %1.2f s \n\n', toc);
      x = x(:,x_indx);                                                    % Picking data inside the invariant set for plot
        plot(xz,yz,'.r')
            hold on
        plot(x(1,:),x(2,:),'.b')
        plot(cent(1,:),cent(2,:),'.k')      
   frame = getframe(1);
      im = frame2im(frame);
      [imind,cm] = rgb2ind(im,256);
      if epoch == 1
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
      else
          imwrite(imind,cm,filename,'gif','WriteMode','append');
      end
      fplot(@(x) sin(x), @(x) cos(x))
      hold off
end


%% ********************************* Plots ********************************
hq = griddata(x(1,:), x(2,:), V, xq, yq);
figure
surf(xq, yq, hq, 'EdgeColor','none')
view(2)
xlabel('$x_1$', 'Interpreter','latex')
ylabel('$x_2$', 'Interpreter','latex', 'Rotation',0, 'HorizontalAlignment','right')
title('$ h_{\max}$', 'Interpreter','latex')
pbaspect([1 1 1])
colorbar
ax = gca;
ax.CLim = [0, 1]; % forces the colorbar to be inside -8 and 2 for better visualization
set(gca,'TickLength',[0.03,0.025], 'layer','top', 'XGrid', 0, 'YGrid', 0, 'LineWidth', 2, 'Box','on');
set(gca, 'fontname', 'Times New Roman', 'fontsize', 40);
HausdorffDist(x',[xz,yz],[],[])
%%
% figure
% ghat=C*phi;
% Xh=x.*(ghat<=1);
% plot(Xh(1,:),Xh(2,:),'.k')
% title('Sublevel of  C^T \phi')
% 
% figure
% plot3(x(1,:), x(2,:), ghat, '.');
% title('C^T \phi')
% save Julia_koopman.mat N_epoch Nh Nlift Nsamp Tresh x g hmax