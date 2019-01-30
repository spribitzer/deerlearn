%% This script prepares the DEER data from Edwards and Stoll (10.1016/j.jmr.2018.01.021) to be used with the python and tensorflow based DEERlearn 
clear 
clc

TimeTraceLength = 256;
PofRLength = 256;

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

nTraces = height(DataTable);
% -------------------------------------------------------------------------
% For testing
% nTests = 16;
% nTraces = nTests;

% prepare tensors
TD = zeros(nTraces,TimeTraceLength);
Tmaxvec = zeros(1,nTraces);

PR = zeros(nTraces,PofRLength);
Rs = zeros(nTraces,PofRLength);

% randomize order of the learn data set
rng(512)
randInd = randperm(nTraces);
% -------------------------------------------------------------------------
% Testing
% randInd = 1:nTraces;

% picks the time domain signal and pairs it with the corresponding P(r)
for i = 1 : nTraces
% -------------------------------------------------------------------------
% For Testing
% for i = 1 : nTests
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


% -------------------------------------------------------------------------
% for testing, displays the P(r)'s and S(t)'s versus their corresponding
% axes as well as per elements. allows to check wether scaling was done
% correctly 
%
% figure(1)
% clf
% hold on
% for i = 4 : 14
%   subplot(2,2,1)
%   hold on
%   plot(Rs(i,:),PR(i,:))
%   
%   subplot(2,2,2)
%   hold on
%   plot(PR(i,:))
%   
%   subplot(2,2,3)
%   hold on
%   tvec = linspace(0,Tmaxvec(i),TimeTraceLength);
%   plot(tvec,TD(i,:))
%   
%   subplot(2,2,4)
%   hold on
%   plot(TD(i,:))
% end
% -------------------------------------------------------------------------


%% write to csv files
csvwrite('PR.csv',PR);
csvwrite('TD.csv',TD);
csvwrite('Rs.csv',Rs);
csvwrite('TmaxVec.csv',Tmaxvec.');
csvwrite('Tref.csv',tref);
