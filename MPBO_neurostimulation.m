
%@file OptimizeNeurostim.m
%@authors   Marco Bonizzato https://github.com/mbonizzato
%           Samuel Laferriere https://github.com/samlaf
%           Rose Guay-Hottin https://github.com/RoseGH20
%@version 1.0 / June 2021


%Full, variable-lengthscales implementation of GP-BO optimization of neurostimulation
%The code below displays algorithmic performance on provided datasets
%For all values of a selected hyperparameter.

%Hyperparameter selection is crucial for GP-BO applications. We recommend
%running this code on own existing or surrogate data to tune at least the
%UCB acquisition function hyperparameter "k" (kappa).

% Select moodality
dataset='nhp';              %selected dataset
which_opt= 'MP';     %hyperparameter to optimize
nRep=30;                    %number of repetitions

q_star=32;

% Load data
if strcmp(dataset,'nhp')
    %nhp dataset has 4 subjects
    load('Macaque1_M1_181212.mat')
    load('Macaque2_M1_190527.mat')
    load('Cebus1_M1_190221.mat')
    load('Cebus2_M1_200123.mat')
    SETS=[Macaque1_M1_181212 Macaque2_M1_190527 Cebus1_M1_190221 Cebus2_M1_200123];
    % Provide a rough estimation of running time
    exp_time=ceil(22*130/50*nRep*[10 16]/60/60);
    
elseif strcmp(dataset,'rat')  
    %rat dataset has 6 subjects  
    load('rat1_M1_190716.mat')
    load('rat2_M1_190617.mat')
    load('rat3_M1_190728.mat')
    load('rat4_M1_191109.mat')
    load('rat5_M1_191112.mat')
    load('rat6_M1_200218.mat')
    SETS=[rat1_M1_190716 rat2_M1_190617 rat3_M1_190728 rat4_M1_191109 rat5_M1_191112 rat6_M1_200218];
    % Provide a rough estimation of running time
    exp_time=ceil(27*35/50*nRep*[10 16]/60/60);
end

if exp_time(2)==1
    disp(['The procedure should take less than 1 hour on a standard workstation (Intel i7-6700 @ 3.4GHz).'])
else    
    disp(['The procedure can take approximately ' num2str(exp_time(1)) ' to ' num2str(exp_time(2)) ' hours on a standard workstation (Intel i7-6700 @ 3.4GHz).'])
end

mKernel=5;                  %Matern kernel order
noise_min= 0.001;           %Non-zero to avoid numerical instability


%rho (high, low) is the kernel geometrical hyperparameter (lengthscales)
%kappa is the UCB acquisition function hyperparameter
%nrand is the number of random queries performed to initialize the GP-BO
%noisemax is the maximum value for the noise hyperparameter
if strcmp(dataset,'nhp') 
    kappa=4;   
    rho_high=3; 
    rho_low=0.01;
    nrnd=1;
    noisemax=0.1;
elseif strcmp(dataset,'rat') 
    kappa=3;
    rho_high=3; 
    rho_low=0.01;
    nrnd=1;
    noisemax=0.1;
end
%kappa is the only parameter we vary between the nhp and rat implementation

%A selection of values to be tested for the selected hyperparameter
if strcmp(which_opt,'rholow')
    this_opt = [0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 2]; % rho_low
elseif strcmp(which_opt,'rhohi')
    this_opt= [1.1 1.2:.3:3 4:10 ]; %rho_high
elseif strcmp(which_opt,'noisemax')
    this_opt=[0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10]; % max noise
elseif strcmp(which_opt,'kappa')
    if strcmp(dataset,'nhp')
        this_opt=[1:.5:2 2.5:.3:5.5 6:10]; % kappa
    else
        this_opt= [1 1.5 2 2.3:.3:4.1 5:10];
    end
elseif strcmp(which_opt,'nrand')
    if strcmp(dataset,'nhp')
        this_opt= [1:5 10:10:90 95 96]; %nrand
    else
        this_opt= [1:5 7 10:5:30 32];
    end
elseif strcmp(which_opt,'MP')  %MP-GPBO
    if strcmp(dataset,'nhp')
        this_opt= [1  1 1]; %MP
    else
        this_opt= [1 1 1];
    end
end
    
%update timing estimation utilities
tic;
count_perf=0;
tot_emgs=0;
max_emgs=0;
for m_i=1:numel(SETS) 
    tot_emgs=tot_emgs+ length(SETS(m_i).emgs);
    if length(SETS(m_i).emgs)>max_emgs
        max_emgs=length(SETS(m_i).emgs);
    end
end
tot_perf=tot_emgs*length(this_opt);

%prepare results storage
hyperparams = cell(size(SETS,2),max_emgs,numel(this_opt),nRep,SETS(1).nChan); %hyperparameters
Stored_MaxSeenResp = cell(size(SETS,2),max_emgs,numel(this_opt),nRep,SETS(1).nChan);  %maximum recorded response
Stored_perf_explore = cell(numel(SETS),max_emgs); %exploration score
Stored_perf_exploit = cell(numel(SETS),max_emgs); %exploitation score
Stored_P_test=cell(numel(SETS),max_emgs,numel(this_opt),nRep); %points tested and response
Stored_MappingAccuracyRSQ=cell(size(SETS,2),max_emgs,numel(this_opt),nRep); %R-squared of mapping accuracy
Stored_YMU = cell(size(SETS,2),max_emgs,numel(this_opt),nRep,SETS(1).nChan); %predicted map, as average of GP fitting

for m_i=1:numel(SETS)                         %for each subject
    
    subject= SETS(m_i);
    
    for k_i=1:numel(this_opt)       %for each hyperparameter value
    
        if strcmp(which_opt,'rholow')
            rho_low=this_opt(k_i);
        elseif strcmp(which_opt,'rhohi')
            rho_high= this_opt(k_i);
        elseif strcmp(which_opt,'noisemax')
            noisemax=this_opt(k_i);
        elseif strcmp(which_opt,'kappa')
            kappa=this_opt(k_i);
        elseif strcmp(which_opt,'nrand')
            nrnd= this_opt(k_i);
        end

        for e_i=1:numel(subject.emgs) % for each muscle of the given subject

            %display remaining time information
            %[m_i k_i syn]
            disp([num2str(count_perf/tot_perf*100) ' % completed'])
            if count_perf>0
                t=toc;                
                hs=floor((t/count_perf)*(tot_perf-count_perf)/60/60);
                mins=floor((t/count_perf)*(tot_perf-count_perf)/60-hs*60);
                disp(['Estimated remaining time: ' num2str(hs) ' hours, ' num2str(mins) ' minutes.'])
            end

            % "Ground truth" map
            MPm=subject.sorted_respMean(:,e_i);
            % Best known channel
            mMPm=max(MPm);

            % Create the kernel
            % Put a box prior on the two lengthscale hyperparameters
            % note that params are in log scale
            priorbox = {@priorSmoothBox1,log(rho_low),log(rho_high),100};
            priorbox2 = {@priorSmoothBox1,log(0.01),log(100),100};
            prior.cov = {priorbox; priorbox; priorbox2};
            infm = @infGaussLik;
            covf = {@covMaternard,mKernel};
            likf = @likGauss;
            prior.lik = {{@priorSmoothBox1, log(noise_min), log(noisemax), 100}}; 
            infprior = {@infPrior,@infGaussLik,prior};

            % Then run the sequential optimization
            perf_explore=[]; %performance
            DimSearchSpace = subject.nChan;
            MaxQueries = DimSearchSpace;
            % Here the maximum number of queries is equal to the maximum
            % number of electrodes, which is the number of queries required
            % for extensive search
            P_test = cell(nRep,1); %storing all queries  
            clear perf_explore perf_exploit perf_rsq

            for rep_i=1:nRep % for each repetition

                passed=0; %extreme hyperparameter values may give rise  
                %occasionally to numerical instabilities. We use a
                %try-catch construct to simply reboot the search attempt in
                %case of numerical problems
                P_max=[];

                while passed==0
                    try

                        
                        MaxSeenResp=0; %maximum response obtained in this round,
                        %used to normalize all responses between zero and one.
                        q_excl=[];
                        q=1; % query number
                        hyp = struct('mean', [], 'cov',log([1 1 1]), 'lik', log(1));  
                        %initialize kernel hyperparameters
                        clear valid_resp

                        order_this=randperm(subject.nChan); %random permutation
                        %of each entry of the search space

                        while q <= MaxQueries  
                           
                            %We will sample the search space randomly for
                            %exactly nrnd queries
                            if q>nrnd 
                                % Find next point (max of acquisition function)
                                AcquisitionMap = fmu + kappa.*real(sqrt(fs2)); %UCB acquisition   
                                NextQuery = find(ismember(AcquisitionMap, max(AcquisitionMap))); 
                                %select next query
                                if length(NextQuery) > 1
                                    NextQuery = NextQuery(randi(numel(NextQuery)));
                                end
                                P_test{rep_i}(q,1) = NextQuery; 
                            else 
                                P_test{rep_i}(q,1) = order_this(q); 
                            end
                            if strcmp(which_opt,'MP')
                                if k_i==3
                                    if q> q_star

                                P_test{rep_i}(q,1) = order_this(q); 
                                    end
                                end
                            end
                            
                            query_elec = P_test{rep_i}(q,1);
                            



                            %This offline optimization code randomly choses one
                            %response among all responses stored in the
                            %selected search space look-up table.
                            valid_resp=subject.sorted_resp{query_elec,e_i}(subject.sorted_isvalid{query_elec,e_i}~=0);
                            r_i=randi(numel(valid_resp));  
                            test_respo= valid_resp(r_i);
                            % done reading response
                            P_test{rep_i}(q,2)=test_respo;
                            %The first element of P_test is the selected search
                            %space point, the second the resulting value

                            if (test_respo>MaxSeenResp) || (MaxSeenResp==0)
                                %updated maximum response obtained in this
                                %round
                                MaxSeenResp=test_respo;
                            end

                            if q> q_star
                                if strcmp(which_opt,'MP')
                                    if k_i==2 %in case of MP-GPBO, drop queries after q star, but never the one of the channel considered best, nor the one you just got
                                        remain_q=find(q_excl);
                                        the_best= find(P_test{rep_i}(q,2))==BestQuery;
                                        q_excl_available=setdiff(remain_q,the_best);
                                        exc_this=randi(length(q_excl_available));
                                        q_excl(q_excl_available(exc_this))=0;
                                    end
                                end
                            end
                            
                            q_excl=[q_excl; 1]; % if 1, the query is available, if 0, it has been dropped

                            if ((q> q_star) && (strcmp(which_opt,'MP')) && (k_i==3) && (test_respo>MaxSeenResp))
                                            P_max(q)= query_elec; 
                            else

                                x = subject.ch2xy(P_test{rep_i}(q_excl==1,1),:); %search space position
                                y = P_test{rep_i}(q_excl==1,2)/MaxSeenResp;      %test result
    
                                %GP-BO optimization
                                evalc('hyp = minimize(hyp, @gp, -10, infprior, [], covf, likf, x, y);');
                                %evalc used to suppress spurious screen output
                                [ymu, ys2, fmu, fs2] = gp(hyp, infm, [], covf, likf, x, y, subject.ch2xy);
    
                                if strcmp(which_opt,'MP')
                                    if k_i==2       %in case of MP-GPBO, recalculate fs2 so to avoid raising uncertainty
                                        for f_i=1:size(fs2,1)
                                            for f_j=2:size(fs2,2)
                                                fs2(f_i,f_j)=min(last_fs2(f_i,f_j),fs2(f_i,f_j));
                                            end
                                        end
                                        last_fs2=fs2;
                                    end
                                end
                                
                                % We only test for gp predictions at electrodes that
                                % we had queried (presumable we only want to return an
                                % electrode that we have already queried).                        
                                Tested=unique(sort(P_test{rep_i}(:,1)));
                                MapPredictionTested=ymu(Tested);
                                BestQuery=Tested(find(ismember(MapPredictionTested, max(MapPredictionTested))));
                                if length(BestQuery) > 1
                                    BestQuery = BestQuery(randi(numel(BestQuery)));
                                end
    
                                %Maximum response at time q
                                P_max(q)= BestQuery; 
                                
                            end
                            %store all info
                            hyperparams{m_i,e_i,k_i,rep_i,q} = hyp; 
                            Stored_MaxSeenResp{m_i,e_i,k_i,rep_i,q} = MaxSeenResp;
                            Stored_YMU{m_i,e_i,k_i,rep_i,q}=ymu;
                            %ymu is the predicted average GP value across the map
                            q=q+1; 
                        end

                        %estimate current exploration performance: 
                        %knowledge of best stimulation point
                        perf_explore(rep_i,:)=MPm(P_max)/mMPm; 
                        %estimate current exploitation performance: 
                        %knowledge of best stimulation point
                        perf_exploit(rep_i,:)= P_test{rep_i,1}(:,1);
                        %calculate model fitting of ground truth value map
                        mdl=corrcoef(MPm,ymu);        
                        perf_rsq(rep_i,:)=mdl(2)^2;

                        %at this point, no numerical instability was detected
                        passed=1;
                    catch    
                        passed=0;    
                        disp('Captured a Cholesky factorization numerical error')    
                    end
                end    

                %store all tests 
                Stored_P_test{m_i,e_i,k_i,rep_i}=P_test{rep_i};
            end

            %store all performance estimations
            Stored_MappingAccuracyRSQ{m_i,e_i,k_i}=perf_rsq;
            Stored_perf_explore{m_i,e_i,k_i}=perf_explore;
            Stored_perf_exploit{m_i,e_i,k_i}= MPm(perf_exploit)/mMPm;
            
            count_perf=count_perf+1;
        end
    end
end

%save all workspace, excluding datasets
dd = datetime('today');
dd1=num2str(year(dd));
dd1=dd1(end-1:end);
dd2=month(dd);
if dd2<10
    dd2=['0' num2str(dd2)];
else
    dd2=[num2str(dd2)];
end
dd3=day(dd);
if dd3<10
    dd3=['0' num2str(dd3)];
else
    dd3=[num2str(dd3)];
end
ddate=[dd1 dd2 dd3];
fn=[dataset '_' which_opt '_' num2str(nRep) '_' ddate]; %file name
save(fn,'covf','dataset','hyperparams','infm','kappa','likf','MaxQueries',...
    'mKernel','Stored_MaxSeenResp','noisemax','noise_min','nRep','nrnd','prior','Stored_perf_explore',...
    'Stored_perf_exploit','Stored_P_test','rho_high','rho_low','Stored_MappingAccuracyRSQ','this_opt','which_opt','Stored_YMU')


%% this makes a graph of blue: GP, red: MP-GPBO
figure
ccc=[];
ccc2=[];
ccc3=[];
for i_i=1:size(Stored_perf_explore,1)
    for j_i=1:size(Stored_perf_explore,2)
        if ~isnan(mean(Stored_perf_explore{i_i,j_i,1}))
        ccc=[ccc; mean(Stored_perf_explore{i_i,j_i,1})];
        ccc2=[ccc2; mean(Stored_perf_explore{i_i,j_i,2})];
        ccc3=[ccc3; mean(Stored_perf_explore{i_i,j_i,3})];

        end
    end
end

subplot(1,2,1)
plot(mean(ccc),'b')
hold on
plot(mean(ccc)+std(ccc)/sqrt(size(ccc,1)),'b')
plot(mean(ccc)-std(ccc)/sqrt(size(ccc,1)),'b')
plot(mean(ccc2),'r')
plot(mean(ccc2)+std(ccc2)/sqrt(size(ccc2,1)),'r')
plot(mean(ccc2)-std(ccc2)/sqrt(size(ccc2,1)),'r')
plot(mean(ccc3),'g')
plot(mean(ccc3)+std(ccc3)/sqrt(size(ccc3,1)),'g')
plot(mean(ccc3)-std(ccc3)/sqrt(size(ccc3,1)),'g')

ylim([0 1])
plot([32 32],[0 1],'k--')


subplot(1,2,2)
plot(mean(ccc),'b')
hold on
plot(mean(ccc2),'r')
plot(mean(ccc3),'g')
plot([32 32],[0 1],'k--')

%% calculates time on x axis
% but doesnt work well

for r_i=1:30
for t_i=1:96
    x = subject.ch2xy(P_test{r_i}(1:t_i,1),:); %search space position
    y = P_test{r_i}(1:t_i,2)/MaxSeenResp;      %test result
    tic
    evalc('hyp = minimize(hyp, @gp, -10, infprior, [], covf, likf, x, y);');
    gp(hyp, infm, [], covf, likf, x, y, subject.ch2xy);
    t=toc;
    times(t_i,r_i)=t;
    x = subject.ch2xy(P_test{r_i}(1:min(t_i,32),1),:); %search space position
    y = P_test{r_i}(1:min(t_i,32),2)/MaxSeenResp;      %test result
    tic
    evalc('hyp = minimize(hyp, @gp, -10, infprior, [], covf, likf, x, y);');
    gp(hyp, infm, [], covf, likf, x, y, subject.ch2xy);
    t=toc;
    times2(t_i,r_i)=t;
end
end


for t_i=1:96
timesS(t_i)=sum(times(1:t_i));
timesS2(t_i)=sum(times2(1:t_i));
end

figure
ccc=[];
ccc2=[];
for i_i=1:size(Stored_perf_explore,1)
    for j_i=1:size(Stored_perf_explore,2)
        if ~isnan(mean(Stored_perf_explore{i_i,j_i,1}))
        ccc=[ccc; mean(Stored_perf_explore{i_i,j_i,1})];
        ccc2=[ccc2; mean(Stored_perf_explore{i_i,j_i,2})];
        end
    end
end

subplot(1,2,1)
plot(mean(ccc),'b')
hold on
plot(mean(ccc)+std(ccc)/sqrt(size(ccc,1)),'b')
plot(mean(ccc)-std(ccc)/sqrt(size(ccc,1)),'b')
plot(mean(ccc2),'r')
plot(mean(ccc2)+std(ccc2)/sqrt(size(ccc2,1)),'r')
plot(mean(ccc2)-std(ccc2)/sqrt(size(ccc2,1)),'r')

ylim([0 1])
plot([32 32],[0 1],'k--')

subplot(1,2,2)
plot(timesS,mean(ccc),'b')
hold on
plot(timesS2, mean(ccc2), 'r')
% plot([32 32],[0 1],'k--')