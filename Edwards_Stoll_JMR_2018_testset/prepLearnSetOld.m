% prep deer data
clear 
clc

PofRs = load('distributions_2LZM');

[nDataSets, traceLenghts] = size(PofRs.P0);

DataPoints = 256;

r0 = linspace(min(PofRs.r0),max(PofRs.r0),DataPoints);

P0 = zeros(nDataSets,DataPoints);

for i = 1 : nDataSets
  P0(i,:) = interp1(PofRs.r0,PofRs.P0(i,:),r0);  
end

dat = load('timetraces_2LZM');
TimeTraces = dat.data;

DataTable = struct2table(TimeTraces);

TraceLength = 161;

subs = and(DataTable.nt == 161,DataTable.tmax == 1.6);

Subset = DataTable(subs,:);

remainingTraces = height(Subset);

randInd = randperm(remainingTraces);

TD = zeros(remainingTraces,TraceLength);
PR = zeros(remainingTraces,DataPoints);


for i = 1 : remainingTraces
  S0 = Subset.S0(randInd(i));
  TD(i,:) = S0{1};
  PR(i,:) = P0(Subset.Pidx(randInd(i)),:);
end

csvwrite('PR.csv',PR);
csvwrite('TD.csv',TD);

