%% MONTE CARLO SIMULATION WITH SAP2000 API AND MATLAB

% The script performs a Monte Carlo simulation of a structural model which is previously defined in Sap2000.
% The script requires as inputs:
% - Sap2000 .sdb model;
% - set of N values of parameters, sampled according to the user-defined probability density function.
% The script computes the response of the structural system for each sample of parameters, and save it in a .mat file. The data can be then retrieved to study uncertainty propagation. 
% Different types of analyses (static, modal, dynamic) can be peformed.
% Warning: the script is written for Sap2000 version 19; for different versions, the script should be changed accordingly (change all "SAP2000v19").

%%
% Clean up the workspace
clear;
clc;
rng(1234); % 
%% SETTINGS
% Number of steps
n_step = [];
% Time step
DT = [];
f_n = [];

% Select the desired number of samples 
N = 2; 
data= cell(1,N);
alpha = unifrnd(0,1,[N,1]);

% Modulus of Elasticity
m = 1.97; v = (0.141*m)^4;
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));
E = exp(lhsnorm(mu,sigma,N))*10^8;

% Yield stress
m = 1; v = (0.1*m)^4;
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));
Fy = exp(lhsnorm(mu,sigma,N))*275000;
Fu = 1.5636 * Fy;
Fue = Fu;
Fye = 1.1 * Fy;

% Strain parameters
strain_max_stress = 0.11;
strain_hardening =  strain_max_stress/7.33;
strain_rupture = 1.5455 * strain_max_stress;
final_slope = -0.1;

% Excitation superpotition coefficient
alpha = unifrnd(0,1,[N,1]);

% Parameters
parameters = [E, Fy, alpha];
%% INITIALIZATION
% Set the program executable path
ProgramPath = 'C:\Program Files\Computers and Structures\SAP2000 19\sap2000.exe';
feature('COM_SafeArraySingleDim', 1);
feature('COM_PassSafeArrayByRef', 1);
% Set the API dll path
APIDLLPath = 'C:\Program Files\Computers and Structures\SAP2000 19\sap2000v19.dll';
% Create OAPI helper object
a = NET.addAssembly(APIDLLPath);
helper = SAP2000v19.Helper;
helper = NET.explicitCast(helper,'SAP2000v19.cHelper');
% Create Sap2000 object
SapObject = helper.CreateObject(ProgramPath);
SapObject = NET.explicitCast(SapObject,'SAP2000v19.cOAPI');
% Start Sap2000 application
SapObject.ApplicationStart;
% Hide Sap2000 application
SapObject.Hide;
% Create SapModel object
SapModel = NET.explicitCast(SapObject.SapModel,'SAP2000v19.cSapModel');
% Initialize model
SapModel.InitializeNewModel;
% Open user defined .sdb model: set FileName as the name of the model
% to be analyzed
File = NET.explicitCast(SapModel.File,'SAP2000v19.cFile');
File.OpenFile([pwd,'\','Model.sdb'])
SapModel.SetPresentUnits(SAP2000v19.eUnits.kN_m_C);
% Unlock the model
SapModel.SetModelIsLocked(false);

for i=1:N
    tic
    % Set the i-th parameter sample E_i to the material 'CONC1'
    % The user has to define in the model FileName.sdb an appropriate material (in this case 'CONC1') whose properties are changed at each loop
    PropMaterial=NET.explicitCast(SapModel.PropMaterial,'SAP2000v19.cPropMaterial');
    PropMaterial.SetMPIsotropic('STEEL', E(i), 0.3, 1.170e-05);
    PropMaterial.SetOSteel_1('STEEL',Fy(i),Fu(i),Fye(i),Fue(i),1,0,strain_hardening,strain_max_stress,strain_rupture,final_slope);
    %PropMaterial.SetWeightAndMass('CONC1',1,1);
    %PropMaterial.SetOSteel_1('test',Fy,Fu,Fye,Fue,1,0,sreain_hardening,strain_max_stress,strain_rupture,final_slope);
    %PropMaterial.SetOConcrete_1('CONC1', Fc', False, 0, 2, 4, StrainAtfc, UltimateStrain, -0.1);
    
    %% RUN ANALYSIS
            %Compute the time history as a linear compination of th_1 and th_2
            th_1 = load('El_Centro_1.txt');
            th_2 = load('El_Centro_2.txt');
            RandomTimeHistory = alpha(i) * th_1 + (1-alpha(i)) * th_2;
            writematrix(RandomTimeHistory, 'temp.txt');
            %Replace comma seperators (',') with void (' ')
            fid = fopen('temp.txt','rt') ;
            X = fread(fid) ;
            fclose(fid) ;
            X = char(X.') ;
            Y = strrep(X, ',', ' ') ;
            fid2 = fopen('RandomTimeHistory.txt','wt') ;
            fwrite(fid2,Y) ;
            fclose (fid2) ;
            %Define the Load Case
            Func = NET.explicitCast(SapModel.Func,'SAP2000v19.cFunction');
            FuncTH = NET.explicitCast(Func.FuncTH,'SAP2000v19.cFunctionTH');
            ret = FuncTH.SetFromFile_1("ElCentro", "RandomTimeHistory.txt", 0, 0, 8, 1, true,0.02,0.02);
            MyLoadType = {"Accel", "Accel"};
            MyLoadName ={"U1","U2"};
            MyFunc = {"ElCentro","ElCentro"};
            MySF = [0.025, 0.025];
            MyTF = [1,1];
            MyAT = [0,0];
            MyCSys = {"Global","Global"};
            MyAng = [0,0];
            LoadCases = NET.explicitCast(SapModel.LoadCases,'SAP2000v19.cLoadCases');
            DirHistNonlinear = NET.explicitCast(LoadCases.DirHistNonlinear,'SAP2000v19.cCaseDirectHistoryNonlinear');            
            ret = DirHistNonlinear.SetLoads("DYNAMIC", 2, MyLoadType, MyLoadName, MyFunc, MySF, MyTF, MyAT, MyCSys, MyAng);     
            % Create model analysis
            Analyze = NET.explicitCast(SapModel.Analyze,'SAP2000v19.cAnalyze');
            % Select model case to run
            Analyze.SetRunCaseFlag('MODAL',false);
            Analyze.SetRunCaseFlag('DYNAMIC',true);
            Analyze.SetRunCaseFlag('STATIC',false);
            % Run analysis
            Analyze.RunAnalysis;
            AnalysisResults = NET.explicitCast(SapModel.Results,'SAP2000v19.cAnalysisResults');
            AnalysisResultsSetup = NET.explicitCast(AnalysisResults.Setup,'SAP2000v19.cAnalysisResultsSetup');
            % Select results cases
            AnalysisResultsSetup.DeselectAllCasesAndCombosForOutput;
            AnalysisResultsSetup.SetCaseSelectedForOutput('DYNAMIC');
            % Set output option for modal history results (2: step-by-step)
            AnalysisResultsSetup.SetOptionDirectHist(2);
            % Create variables            
            NumberResults = 0;
            Obj = NET.createArray('System.String', 1);
            Elm = NET.createArray('System.String', 1);
            LoadCase = NET.createArray('System.String', 1);
            StepType = NET.createArray('System.String', 1);
            StepNum = NET.createArray('System.Double', 1);
            U1 = NET.createArray('System.Double', 1);
            U2 = NET.createArray('System.Double', 1);
            U3 = NET.createArray('System.Double', 1);
            R1 = NET.createArray('System.Double', 1);
            R2 = NET.createArray('System.Double', 1);
            R3 = NET.createArray('System.Double', 1);
            % Get model response (displacements and rotations) at all nodes
            [~,~,~,Elm,~,~,~,U1,U2,U3,R1,R2,R3]  = AnalysisResults.JointDispl('ALL', SAP2000v19.eItemTypeElm.GroupElm, NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3);
            
            if i==1
                % Get time step (DT) and number of time steps (n_step)
                [ret,n_step,DT] = DirHistNonlinear.GetTimeStep('DYNAMIC',1,1);
                nStep = n_step+1;
                
                % Get number of model nodes
                PointObj = NET.explicitCast(SapModel.PointObj,'SAP2000v19.cPointObj');
                n_joints = PointObj.Count();
                % Save variables
                Eelm = cell(Elm.Length/nStep,1);
                jj=1;
                for j=1:nStep:Elm.Length
                    Eelm{jj,1}=char(Elm(j));
                    jj=jj+1;
                end
                % Save nodes labelling
                Eelm_ = repmat(Eelm,1,nStep);
                joints(:,:,1) = strcat(Eelm_,'_U1');
                joints(:,:,2) = strcat(Eelm_,'_U2');
                joints(:,:,3) = strcat(Eelm_,'_U3');
                joints(:,:,4) = strcat(Eelm_,'_R1');
                joints(:,:,5) = strcat(Eelm_,'_R2');
                joints(:,:,6) = strcat(Eelm_,'_R3');
                % Convert to categorical variable to save space
                joints=categorical(joints);
            end
            for j=1:n_joints
                    [ret,x,y,z]=PointObj.GetCoordCartesian(Eelm{j,1}, 0, 0, 0);
                    xyz(j,:)=[x,y,z];
            end
            u(:,:,1) = reshape((U1.double)',[n_joints,nStep]);
            u(:,:,2) = reshape((U2.double)',[n_joints,nStep]);
            u(:,:,3) = reshape((U3.double)',[n_joints,nStep]);
            u(:,:,4) = reshape((R1.double)',[n_joints,nStep]);
            u(:,:,5) = reshape((R2.double)',[n_joints,nStep]);
            u(:,:,6) = reshape((R3.double)',[n_joints,nStep]);
            SapModel.SetModelIsLocked(false);
            toc
 
   %Save data in the appropriate format 
   [nodes, Nt, dofs] = size(u);
   u2 = reshape(u,[nodes*Nt,dofs]);
   dim = size(u2);
   dim = dim(1);
   u_new = zeros(dofs*nodes,Nt);
   
   for k = 1:dim
       node_k = ceil(k/Nt);
       Nt_k = k - (node_k-1) * Nt;
       u_new((node_k-1)*dofs + 1 : node_k*dofs, Nt_k) = u2(k,:);
   end
   
   data{1,i}= u_new;
end
      
data = cell2mat(data);
N_dofs = size(data); N_dofs = N_dofs(1);
data = reshape(data,[N_dofs ,Nt, N]);
data = permute(data,[3,2,1]);

%Obtain free dofs
for i=1:N
    u_new = reshape(data(i,:,:),[Nt, N_dofs]);
    counter = 0;
    for j=1:N_dofs
        if sum(u_new(:,j)) == 0
        counter = counter + 1;
        fixed_dofs(counter) = j;
        end
    end
    u_new(:,fixed_dofs) =[];
    data_new(i,:,:) = u_new;
    N_dofs_free = size(data_new); N_dofs_free = N_dofs_free(3);
end

%split data according to its dof
for sample=1:N  
counter_ux=0; counter_uy=0; counter_uz=0;
counter_rx=0; counter_ry=0; counter_rz=0;
    for i=1:N_dofs_free
        if rem(i-1, 6) == 0
            counter_ux = counter_ux + 1;
            Ux(sample,:,counter_ux) = data(sample, :, i);
        end
        if rem(i-2, 6) == 0
            counter_uy = counter_uy + 1;
            Uy(sample,:,counter_uy) = data(sample, :, i);
        end
          if rem(i-3, 6) == 0
            counter_uz = counter_uz + 1;
            Uz(sample,:,counter_uz) = data(sample, :, i);
          end
          if rem(i-4, 6) == 0
            counter_rx = counter_rx + 1;
            Rx(sample,:,counter_rx) = data(sample, :, i);
          end
          if rem(i-5, 6) == 0
            counter_ry = counter_ry + 1;
            Ry(sample,:,counter_ry) = data(sample, :, i);
          end
          if rem(i-6, 6) == 0
            counter_rz = counter_rz + 1;
            Rz(sample,:,counter_rz) = data(sample, :, i);
          end
    end
end

% Save the solutions and the corresponding parameter values (E, Fy, alpha)
save('Ux.mat', 'Ux'); save('Uy.mat', 'Uy');save('Uz.mat', 'Uz')
save('Rx.mat', 'Rx'); save('Ry.mat', 'Ry');save('Rz.mat', 'Rz')
save('parameters.mat', 'parameters');
%Close application
SapObject.ApplicationExit(false());
