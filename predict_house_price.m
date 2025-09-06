% predict_house_price.m
% House price prediction (Mumbai) using a simple MATLAB neural network (Deep Learning Toolbox)
% Save this file as predict_house_price.m and run in MATLAB.

%% Step 0: Notes
% - Place 'house_prices_mumbai.csv' in the same folder as this script or select it via the file dialog.
% - This script expects columns similar to: Location, Latitude, Longitude, SquareFootage, Bedrooms, Bathrooms, Price
% - The script uses fuzzy matching to find column names and trains a simple feedforward network.
% - Requires MATLAB with Deep Learning Toolbox.

%% Step 1: Load Dataset
fileName = 'house_prices_mumbai.csv';
if exist(fileName, 'file') ~= 2
    [file, path] = uigetfile('*.csv', 'Select the house prices dataset');
    if isequal(file,0)
        error('No file selected. Please select a valid dataset.');
    end
    fileName = fullfile(path, file);
end

% Read the dataset
data = readtable(fileName);

% Display available column names
disp('Available columns in dataset:');
disp(data.Properties.VariableNames);

% Find correct column names using fuzzy matching
squareFootCol = findClosestColumn(data.Properties.VariableNames, 'SquareFootage');
bedroomsCol = findClosestColumn(data.Properties.VariableNames, 'Bedrooms');
bathroomsCol = findClosestColumn(data.Properties.VariableNames, 'Bathrooms');
priceCol = findClosestColumn(data.Properties.VariableNames, 'Price');

if isempty(bathroomsCol)
    error('Column "Bathrooms" not found. Please check dataset column names.');
else
    fprintf('Using column "%s" instead of "Bathrooms".\n', bathroomsCol);
end

% Convert Location to Categorical and One-Hot Encode
if ~ismember('Location', data.Properties.VariableNames)
    error('Dataset must contain a ''Location'' column.');
end
locations = unique(data.Location, 'stable');
locationDummy = dummyvar(categorical(data.Location, locations));

% Prepare feature matrix
X = [locationDummy, data.(squareFootCol), data.(bedroomsCol), data.(bathroomsCol)];
y = data.(priceCol);

% Store latitude & longitude (if available)
if ismember('Latitude', data.Properties.VariableNames) && ismember('Longitude', data.Properties.VariableNames)
    latitudes = data.Latitude;
    longitudes = data.Longitude;
else
    latitudes = zeros(height(data),1);
    longitudes = zeros(height(data),1);
end

% Normalize features
X_mean = mean(X, 1);
X_std = std(X, 1);
X = (X - X_mean) ./ X_std;

% Normalize target variable (y)
y_mean = mean(y);
y_std = std(y);
y_normalized = (y - y_mean) / y_std;

%% Step 2: Define Neural Network
inputSize = size(X,2);
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(8)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto');

%% Step 3: Train the Model
net = trainNetwork(X, y_normalized, layers, options);

%% Step 4: User Input for Prediction
userLocation = input('Enter Location: ', 's');
sqft = input('Enter square footage: ');
bed = input('Enter number of bedrooms: ');
bath = input('Enter number of bathrooms: ');

% Find Closest Location Match
matchIdx = find(strcmpi(locations, userLocation), 1);
if isempty(matchIdx)
    warning('Unknown location! Finding best match...');
    distances = cellfun(@(x) levenshteinDistance(lower(userLocation), lower(x)), locations);
    [~, bestMatchIdx] = min(distances);
    userLocation = locations{bestMatchIdx};
    fprintf('Using closest match: %s\n', userLocation);
    matchIdx = bestMatchIdx;
end

% Retrieve Latitude & Longitude (use first matching row from dataset)
userLat = latitudes(matchIdx);
userLon = longitudes(matchIdx);

% Convert location to one-hot encoding
userLocVec = zeros(1, numel(locations));
userLocVec(matchIdx) = 1;

% Prepare user input & normalize
userX = [userLocVec, sqft, bed, bath];
userX = (userX - X_mean) ./ X_std;

% Predict normalized price
predictedNormalizedPrice = predict(net, userX);

% Denormalize the predicted price
y_pred = predictedNormalizedPrice * y_std + y_mean;

% Convert to INR (if dataset is in USD)
isPriceInUSD = false; % <- Set this to true if your data is in USD
exchangeRate = 85.29;

if isPriceInUSD
    predictedPriceINR = y_pred * exchangeRate;
else
    predictedPriceINR = y_pred;
end

% Format price in Lakhs/Crores
formattedPrice = formatINR(predictedPriceINR);

fprintf('\nEstimated House Price in %s: ₹%s\n', userLocation, formattedPrice);

%% Step 5: Display Map with Location
figure;
geoscatter(userLat, userLon, 100, 'filled', 'r');
geolimits([userLat-0.05 userLat+0.05], [userLon-0.05 userLon+0.05]);
geobasemap streets;
title(['Map View: ', userLocation]);

%% Helper Function: Find Closest Column Name
function bestMatch = findClosestColumn(columnNames, target)
    distances = cellfun(@(x) levenshteinDistance(lower(x), lower(target)), columnNames);
    [minDist, idx] = min(distances);
    if minDist <= 3
        bestMatch = columnNames{idx};
    else
        bestMatch = '';
    end
end

%% Levenshtein Distance Function
function dist = levenshteinDistance(s1, s2)
    len1 = length(s1);
    len2 = length(s2);
    dp = zeros(len1+1, len2+1);
    for i = 1:len1+1
        dp(i,1) = i-1;
    end
    for j = 1:len2+1
        dp(1,j) = j-1;
    end
    for i = 2:len1+1
        for j = 2:len2+1
            cost = (s1(i-1) ~= s2(j-1));
            dp(i,j) = min([dp(i-1,j)+1, dp(i,j-1)+1, dp(i-1,j-1)+cost]);
        end
    end
    dist = dp(end,end);
end

%% Format Price to Lakhs/Cr
function formatted = formatINR(price)
    if price >= 1e7
        formatted = sprintf('%.2f Cr', price / 1e7);
    elseif price >= 1e5
        formatted = sprintf('%.2f Lakhs', price / 1e5);
    else
        formatted = sprintf('%.2f', price);
    end
end
