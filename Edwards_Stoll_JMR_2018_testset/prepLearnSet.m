%% This script prepares the DEER data from Edwards and Stoll (10.1016/j.jmr.2018.01.021) to be used with the python and tensorflow based DEERlearn
clear
clc

TimeTraceLength = 256;
PofRLength = 256;

Test = true; % if you set this to true, you will be presented with a plot
% that shows a selected range of S(t)'s and their
% corresponding P(r)'s. both are plotted twice, once with
% their corresponding t/r axis and once without (just the 256
% data points). This allows to verify that the stretching
% process worked correctly
TestStart = 4;
TestEnd = 14;


%% Get time domain signals and prepare them
dat = load('timetraces_2LZM');
TimeTraces = dat.data;

DataTable = struct2table(TimeTraces); % convert data set to a table

TraceLengths = unique(DataTable.tmax);
disp('The following trace lengths (in us) are available:')
disp(TraceLengths)

tref = TraceLengths(2); % this is important, as it sets the reference for what tmax corresponds to the [0.5 7.2] nm r range

%% Get P(r)'s
PofRs = load('distributions_2LZM');
[nDataSets, traceLengths] = size(PofRs.P0);

% resample r and P(r)
r0 = PofRs.r0;
P0 = PofRs.P0;

%% Prepare Output

% -------------------------------------------------------------------------
if Test
  nTests = TestEnd - TestStart;
  nTraces = nTests;
  randInd = TestStart:TestEnd;
else
  nTraces = height(DataTable);
  
  % randomize order of the learn data set
  rng(512)
  randInd = randperm(nTraces);
end

% prepare tensors
TD = zeros(nTraces,TimeTraceLength);
Tmaxvec = zeros(1,nTraces);

PR = zeros(nTraces,PofRLength);
Rs = zeros(nTraces,PofRLength);

LoopLength = nTraces;

% picks the time domain signal and pairs it with the corresponding P(r)
for i = 1 : LoopLength
  index = randInd(i);
  
  S0 = DataTable.S0(index);
  S0 = S0{1};
  tmax = DataTable.tmax(index);
  
  PreppedSignal = prepDEERSignal(S0, TimeTraceLength);
  TD(i,:) = PreppedSignal;
  Tmaxvec(i) = tmax;
  
  Pindex = DataTable.Pidx(index);
  
  [PreppedR, PreppedPofR] = prepPofR(r0,P0(Pindex,:),tmax,tref,PofRLength);
  
  PR(i,:) = PreppedPofR;
  Rs(i,:) = PreppedR;
end


%% plot or write to csv files
% -------------------------------------------------------------------------
% for testing, displays the P(r)'s and S(t)'s versus their corresponding
% axes as well as per elements. allows to check wether scaling was done
% correctly
%
if Test
  figure(1)
  clf
  subplot(2,2,1)
  hold on
  xlabel('r [nm]')
  
  % plot the original P(r) before any interpolation
  Pindex = DataTable.Pidx(TestStart);
  plot(r0,P0(Pindex,:),'k')
  
  subplot(2,2,2)
  hold on
  xlabel('DataPoints')
  
  subplot(2,2,3)
  hold on
  xlabel('t [\mus]')
  
  % plot the original S(t) before any interpolation
  S0 = DataTable.S0(TestStart);
  S0 = S0{1};
  tmax = DataTable.tmax(TestStart);
  taxis = linspace(0,tmax,length(S0));
  plot(taxis,S0,'k')
  
  subplot(2,2,4)
  hold on
  xlabel('DataPoints')
  
  for i = 1 : nTraces
    subplot(2,2,1)
    plot(Rs(i,:),PR(i,:))
    
    subplot(2,2,2)
    plot(PR(i,:))
    
    subplot(2,2,3)
    tvec = linspace(0,Tmaxvec(i),TimeTraceLength);
    plot(tvec,TD(i,:))
    
    subplot(2,2,4)
    plot(TD(i,:))
  end
    
else
  csvwrite('PR.csv',PR);
  csvwrite('TD.csv',TD);
  csvwrite('Rs.csv',Rs);
  csvwrite('TmaxVec.csv',Tmaxvec.');
  csvwrite('Tref.csv',tref);
end
