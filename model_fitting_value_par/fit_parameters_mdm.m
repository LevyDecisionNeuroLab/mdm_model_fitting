% NOTE: Requires MATLAB optim library

% run this file to fit all possible models to each individual subject
% model fitting results saved in MLE structures
% subjective ratings are also saved in the *_fitpar.mat

clearvars
close all

% poolobj = parpool('local', 8);

%% Define conditions
fitparwave = 'Behavior data fitpar_02022021'; % folder to save all the fitpar data structures
fitbywhat = 'value'; % what to use as values 'value', 'rating', 'arbitrary'(0,1,2,3,4)
model = 'ambigOnly'; % which utility function 'ambigNriskValPar', 'ambigSVPar', 'ambigOnly'
includeAmbig = true;
search = 'grid'; % 'grid', 'single'

%% set up fitting parameters

% lower and upper bound
lb_num = 1e-2;
ub_num = 100;

if strcmp(model, 'ambigSVPar')
    lb = [-100 -10 lb_num lb_num lb_num lb_num]; % slope, beta, v1, v2, v3, v4
    ub = [100 10 ub_num ub_num ub_num ub_num];
elseif strcmp(model, 'ambigOnly')
    lb = [-100 -10]; % slope, beta
    ub = [100 10];        
end

% value start point
value_start = 50;

% grid search
grid_step = 0.5;
val_step = 30;

if strcmp(search, 'grid')
    % grid search
    % range of each parameter    
    slopeRange = -4:grid_step:1;
    bRange = -2:grid_step:2;
    aRange = 0:grid_step:4;
    val1Range = 0:val_step:ub_num;
    val2Range = 0:val_step:ub_num;
    val3Range = 0:val_step:ub_num;
    val4Range = 0:val_step:ub_num;
    
%     val1Range = lb_num + rand(1,3) .* (ub_num - lb_num);
%     val2Range = lb_num + rand(1,3) .* (ub_num - lb_num);
%     val3Range = lb_num + rand(1,3) .* (ub_num - lb_num);
%     val4Range = lb_num + rand(1,3) .* (ub_num - lb_num);
    
    if strcmp(model,'ambigNriskValPar')
        [b1, b2, b3, b4, b5, b6, b7] = ndgrid(slopeRange, bRange, aRange, val1Range(2:end), val2Range(2:end), val3Range(2:end), val4Range(2:end));
        % all posibile combinatinos of parameters
        b0 = [b1(:) b2(:) b3(:) b4(:) b5(:) b6(:) b7(:)];
    elseif strcmp(model, 'ambigSVPar')
        [b1, b2, b3, b4, b5, b6] = ndgrid(slopeRange, bRange, val1Range(2:end), val2Range(2:end), val3Range(2:end), val4Range(2:end));
        % all posibile combinatinos of parameters
        b0 = [b1(:) b2(:) b3(:) b4(:) b5(:) b6(:)];% lower and upper bound of fitting parameters
        lb = [-100 -10 lb_num lb_num lb_num lb_num]; % slope, beta, v1, v2, v3, v4
        ub = [100 10 ub_num ub_num ub_num ub_num];
    elseif strcmp(model, 'ambigOnly')
        [b1, b2] = ndgrid(slopeRange, bRange);
        b0 = [b1(:) b2(:)];
        lb = [-100 -10]; % slope, beta
        ub = [100 10];        
    end
elseif strcmp(search,'single')
    if strcmp(model,'ambigNriskValPar')
        % single search
        b0 = [-1 0.5 0.5 value_start value_start value_start value_start]; % starting point of the search process, [gamma, beta, alpha, val1, val2, val3, val4]
    elseif strcmp(model, 'ambigSVPar')
        % single search
        b0 = [-1 0.5 value_start value_start value_start value_start]; % starting point of the search process, [gamma, beta, val1, val2, val3, val4]
    elseif strcmp(model, 'ambigOnly')
        b0 = [-1 0.5];
    end
end

% all values
vals = [5,8,12,25];

base = 0; % another parm in the model. Not used.

fixed_ambig = 0;
fixed_valueP = 5; % Value of fixed reward
fixed_prob = 1;   % prb of fixed reward 

%% Set up loading & subject selection
root = 'E:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis';
data_path = fullfile(root, 'PTB Behavior Log/'); % root of folders is sufficient
rating_filename = fullfile(root, 'Behavior Analysis/MDM_Rating_for_fitting.csv');
fitpar_out_path = fullfile(root, 'Behavior fitpar files',fitparwave);

% if folder does not exist, create folder
if exist(fitpar_out_path)==0
    mkdir(fullfile(root, 'Behavior fitpar files'),fitparwave)
end

addpath(genpath(data_path)); % generate path for all the subject data folder

% get all subjects number in the folder
subjects = getSubjectsInDir(data_path, 'subj');
exclude = [2581]; % TEMPORARY: subjects incomplete data (that the script is not ready for)
subjects = subjects(~ismember(subjects, exclude));
% subjects = [2654 2655 2656 2657 2658 2659 2660 2661 2662 2663 2664 2665 2666];
% subjects = [2663];
% subjects = [2073 2582 2587 2597 2651 2663 2665 2666];
% subjects = [2073 2582 2587 2597 2651 2663 2665 2666 2550 2585 2596 2600 2655 2659 2660 2664];

% load subjective ratings
% column1-subj ID, c2-$0, c3-$5,c4-$8,c5-$12,c6-$25,c7-no effect, c8-slight,c9-moderate,c10-major,c11-recovery.
rating = csvread(rating_filename,1,0); %reads data from the file starting at row offset R1 and column offset C1. For example, the offsets R1=0, C1=0 specify the first value in the file.

%% Individual subject fitting
tic

parfor subj_idx = 1:length(subjects)
  domains = {'MON', 'MED'};

  for domain_idx = 1:length(domains)
    
    
    subjectNum = subjects(subj_idx);
    domain = domains{domain_idx};
    
    % directly use load/save violate transparency, use a function instead 
    Data = load_mat(subjectNum, domain);
    
  %% Load subjective ratings
    % prepare subjective rating for each trial
    if strcmp(domain, 'MON') ==1 % Monetary block
        subjRefRatings = rating(rating(:,1)==subjectNum,3) * ones(length(Data.choice), 1);
        %values = Data.vals(include_indices);
        subjRatings = ones(length(Data.vals),1);
        for i=1:length(subjRatings)
            subjRatings(i) = rating(rating(:,1)==subjectNum,1+find(rating(1,2:6)==Data.vals(i)));
        end
    else % Medical block
        subjRefRatings = rating(rating(:,1)==subjectNum,8) * ones(length(Data.choice), 1);
        %values = Data.vals(include_indices);
        subjRatings = ones(length(Data.vals),1);
        for i=1:length(subjRatings)
            subjRatings(i) = rating(rating(:,1)==subjectNum,6+find(rating(1,7:11)==Data.vals(i)));
        end
    end
    
    %% Refine variables
    
    if includeAmbig
        % Exclude non-responses
        include_indices = Data.choice ~= 0;
    else
        % Exclude ambiguious trials (fit only risky trials)
        include_indices = Data.ambigs' ~= 0 & Data.choice ~= 0;
    end

    choice = Data.choice(include_indices);
    values = Data.vals(include_indices);
    ambigs = Data.ambigs(include_indices);
    probs  = Data.probs(include_indices);
    ratings = subjRatings(include_indices);
    refRatings = subjRefRatings(include_indices);

    
    % Side with lottery is counterbalanced across subjects 
    % code 0 as reference choice, 1 as lottery choice
    % if sum(choice == 2) > 0 % Only if choice has not been recoded yet. RJ-Not necessary
    % RJ-If subject do not press 2 at all, the above if condition is problematic
      if Data.refSide == 2
          choice(choice == 2) = 0;
          choice(choice == 1) = 1;
      elseif Data.refSide == 1 % Careful: rerunning this part will make all choices 0
          choice(choice == 1) = 0;
          choice(choice == 2) = 1;
      end
    
    % choice data for $5 only, for rationality check only
    idx_only5 = and(Data.choice ~= 0, Data.vals' == 5);
    choice5 = Data.choice(idx_only5);
    values5 = Data.vals(idx_only5);
    ambigs5 = Data.ambigs(idx_only5);
    probs5  = Data.probs(idx_only5);
    
    if Data.refSide == 2
        choice5(choice5 == 2) = 0;
        choice5(choice5 == 1) = 1;
    elseif Data.refSide == 1 % Careful: rerunning this part will make all choices 0
        choice5(choice5 == 1) = 0;
        choice5(choice5 == 2) = 1;
    end
    
    choice_prob_5= sum(choice5)/length(choice5);
    
    %% Fitting 
  
    if strcmp(model, 'ambigOnly')
        fitVal = ratings; % in this model, use ratings as the subjective values into the model
        fitrefVal = refRatings;
    else
        fitrefVal = fixed_valueP * ones(length(choice), 1);
        fitVal = values;
    end
    
    % fit the model
    refProb = fixed_prob  * ones(length(choice), 1);

    ambig = unique(ambigs(ambigs > 0)); % All non-zero ambiguity levels 
    prob = unique(probs); % All probability levels

    % Two versions of function:
    %       fit_ambgiNriskValPar_model: unconstrained
    %       fit_ambigNriskValPar_model_Constrained

    % Unconstrained fitting
    % choice dimension 1 by n, ambigs/probs/vals dim n by 1. for model
    % fitting to work need all 1 by n
%     [info, p] = fit_ambigNriskValPar_model_constrained(choice, ...
%         fitrefVal', ...
%         fitVal', ...
%         refProb', ...
%         probs', ...
%         ambigs', ...
%         model, ...
%         b0, ...
%         base, ...
%         vals);
    
    if strcmp(model, 'ambigSVPar')
        [info, p] = fit_ambigNriskValPar_model(choice, ...
            fitrefVal', ...
            fitVal', ...
            refProb', ...
            probs', ...
            ambigs', ...
            model, ...
            b0, ...
            lb,...
            ub,...
            base, ...
            vals);    

        disp(['Subject ' num2str(subjectNum) ' domain' domain ' ambigSVPar fitting completed'])
    elseif strcmp(model, 'ambigOnly')
       [info, p] = fit_ambigOnly_model(choice, ...
            fitrefVal', ...
            fitVal', ...
            refProb', ...
            probs', ...
            ambigs', ...
            model, ...
            b0, ...
            base);    

        disp(['Subject ' num2str(subjectNum) ' domain' domain ' ambigOnly fitting completed'])
        
    end
    
    if strcmp(model, 'ambigNriskValPar')
        slope = info.b(1);
        a = info.b(3);
        b = info.b(2);
        r2 = info.r2;
    elseif strcmp(model, 'ambigSVPar') || strcmp(model, 'ambigOnly')
        slope = info.b(1);
        b = info.b(2);
        r2 = info.r2;      
    end

    % choice probability for each trial based on fitted model parameters
    % should not using the model fitting inputs, but rather also
    % include missing response trials. So IMPORTANTLY, use all trials!
%     choiceModeled = choice_prob_ambigNriskValPar(base,fixed_valueP * ones(length(Data.vals), 1)',Data.vals',...
%         fixed_prob  * ones(length(Data.vals), 1)',Data.probs',Data.ambigs',info.b,model,vals);         

    %% Choice 
    
    % All choices
    choiceAll = Data.choice;
    valuesAll = Data.vals;
    refValue = 5;
    ambigsAll = Data.ambigs;
    probsAll  = Data.probs;
    % mark miss-response
    choiceAll(choiceAll==0) = NaN;

    % Side with lottery is counterbalanced across subjects 
    % code 0 as reference choice, 1 as lottery choice
    % if sum(choice == 2) > 0 % Only if choice has not been recoded yet. RJ-Not necessary
    % RJ-If subject do not press 2 at all, the above if condition is problematic
      if Data.refSide == 2
          choiceAll(choiceAll == 2) = 0;
          choiceAll(choiceAll == 1) = 1;
      elseif Data.refSide == 1 % Careful: rerunning this part will make all choices 0
          choiceAll(choiceAll == 1) = 0;
          choiceAll(choiceAll == 2) = 1;
      end
    
    %% Create choice matrices
    % One matrix per condition. Matrix values are binary (0 for sure
    % choice, 1 for lottery). Matrix dimensions are prob/ambig-level
    % x payoff values. Used for graphing and some Excel exports.
 
    choiceMatrix = create_choice_matrix(values,ambigs,probs,choice);

    %% Graph
%    colors =   [255 0 0;
%     180 0 0;
%     130 0 0;
%     52 181 233;
%     7 137 247;
%     3 85 155;
%     ]/255;
% 
%     figure    
%     counter=5;
%     for i=1:3
%         subplot(3,2,counter)
%         plot(valueP,ambigChoicesP(i,:),'--*','Color',colors(3+i,:))
%         legend([num2str(ambig(i)) ' ambiguity'])
%         if counter==1
%             title(['Beta = ' num2str(b_uncstr)])
%         end
%         ylabel('Chose Lottery')
%         if counter==5
%         xlabel('Lottery Value ($)')
%         end
%         counter=counter-2;
%     end
% 
%     counter=2;
%     for i=1:3
%         subplot(3,2,counter)
%         plot(valueP,riskyChoicesP(i,:),'--*','Color',colors(i,:))
%         legend([num2str(prob(i)) ' probability'])
%         if counter==2
%             title(['Alpha = ' num2str(a_uncstr)])
%         end
%             if counter==6
%         xlabel('Lottery Value ($)')
%             end
%         counter=counter+2;
%     end
% 
%     set(gcf,'color','w');
%     figName=['RA_GAINS_' num2str(subjectNum) '_fitpar'];
% %     exportfig(gcf,figName,'Format','eps','bounds','tight','color','rgb','LockAxes',1,'FontMode','scaled','FontSize',1,'Width',4,'Height',2,'Reference',gca);


%% graph with fitted lines
% 
%     xP = 0:0.1:max(valueP);
%     uFP = fixed_prob * (fixed_valueP).^a_uncstr;
%      
%    figure
%      
%     % risk pos
%     for i = 1 :length(prob)
%         plot(valueP,riskyChoicesP(i,:),'o','MarkerSize',8,'MarkerEdgeColor',colors([1 1 1])...
%             ,'MarkerFaceColor',colors(i,:),'Color',colors(i,:));
%           hold on
%         % logistic function
%         uA = prob(i) * xP.^a_uncstr;
%         p = 1 ./ (1 + exp(slope_uncstr*(uA-uFP)));
% 
%         plot(xP,p,'-','LineWidth',4,'Color',colors(i,:));
%         axis([0 25 0 1])
%         set(gca, 'ytick', [0 0.5 1])
%         set(gca,'xtick', [0 5 10 15 20 25])
%         set(gca,'FontSize',25)
%         set(gca,'LineWidth',3)
%         set(gca, 'Box','off')
% 
% 
%     end
% %     title(['  alpha gain = ' num2str(a_uncstr)]);
%     
%     figure
%     % ambig pos
%     for i = 1:length(ambig)
%         plot(valueP,ambigChoicesP(i,:),'o','MarkerSize',8,'MarkerEdgeColor',colors([1 1 1]),'MarkerFaceColor',colors(length(prob)+i,:));
%          hold on
% % 
%         % logistic function
%         uA = (0.5 - b_uncstr.*ambig(i)./2) * xP.^a_uncstr;
%         p = 1 ./ (1 + exp(slope_uncstr*(uA-uFP)));
% 
% 
%         plot(xP,p,'-','LineWidth',2,'Color',colors(length(prob)+i,:));
%         axis([0 25 0 1])
%         set(gca, 'ytick', [0 0.5 1])
%         set(gca,'xtick', [0 5 10 15 20 25])
%         set(gca,'FontSize',25)
%         set(gca,'LineWidth',3)
%         set(gca, 'Box','off')
% 
%     end
% %     title([ '  beta gain = ' num2str(b_uncstr)]);

    %% Save generated values
    Data.choiceMatrix = choiceMatrix;
    Data.choiceProb5 = choice_prob_5;
    
    % choices per each trial, 0-ref,1-lottery
    Data.choiceLott = choiceAll;
%     Data.choiceModeled = choiceModeled;
    
    if strcmp(model, 'ambigNriskValPar')
        Data.MLE = info;
        Data.alpha = info.b(3);
        Data.beta = info.b(2);
        Data.gamma = info.b(1);
        Data.val_par = info.b(4:7);
        Data.r2 = info.r2;
    elseif strcmp(model, 'ambigSVPar')
        Data.MLE = info;
        Data.beta = info.b(2);
        Data.gamma = info.b(1);
        Data.val_par = info.b(3:6);
        Data.r2 = info.r2;
     elseif strcmp(model, 'ambigOnly')
        Data.MLE = info;
        Data.beta = info.b(2);
        Data.gamma = info.b(1);
        Data.r2 = info.r2;       
    end
    
    % save data struct for the two domains
    % directly using load/save violates transparency, use a function
    % instead
    save_mat(Data, subjectNum, domain, fitpar_out_path)
  end
end

toc 

delete(poolobj)
