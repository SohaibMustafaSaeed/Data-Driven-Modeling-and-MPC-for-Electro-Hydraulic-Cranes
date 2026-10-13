function [ok, weights, config] = validate_controller_package(packageDirectory, verbose)
if nargin < 1
    packageDirectory = fileparts(mfilename('fullpath'));
end
if nargin < 2
    verbose = true;
end
ok = false;
try
    metadata = jsondecode(fileread(fullfile(packageDirectory, 'controller_config.json')));
    for j = 1:numel(metadata.required_runtime_files)
        require(exist(fullfile(packageDirectory, metadata.required_runtime_files{j}), 'file') == 2, ...
            ['Missing file: ' metadata.required_runtime_files{j}]);
    end
    require(strcmp(controller_file_sha256(fullfile(packageDirectory, 'controller_weights.mat')), metadata.weights_mat_sha256), 'Weights MAT hash mismatch.');
    require(strcmp(controller_file_sha256(fullfile(packageDirectory, 'controller_config.mat')), metadata.config_mat_sha256), 'Configuration MAT hash mismatch.');
    require(strcmp(controller_file_sha256(fullfile(packageDirectory, 'feature_schema.json')), metadata.packaged_schema_sha256), 'Schema hash mismatch.');
    weights = load(fullfile(packageDirectory, 'controller_weights.mat'));
    config = load(fullfile(packageDirectory, 'controller_config.mat'));
    names = {'gru_weight_ih','gru_weight_hh','gru_bias_ih','gru_bias_hh','head_weight_1','head_bias_1','head_weight_out','head_bias_out'};
    sizes = {[96 32],[96 32],[96 1],[96 1],[16 32],[16 1],[2 16],[2 1]};
    require(numel(fieldnames(weights)) == numel(names), 'Unexpected network arrays.');
    count = 0;
    for j = 1:numel(names)
        value = weights.(names{j});
        require(isa(value, 'single') && isreal(value) && isequal(size(value), sizes{j}) && all(isfinite(value(:))), ['Invalid weight/bias: ' names{j}]);
        count = count + numel(value);
    end
    require(count == 6898 && config.parameter_count == 6898 && metadata.parameter_count == 6898, 'Parameter count mismatch.');
    require(config.input_size == 32 && config.state_size == 22 && config.hidden_size == 32 && config.head_size == 16 && config.output_size == 2, 'Network dimension mismatch.');
    require(isa(config.input_mean, 'double') && isequal(size(config.input_mean), [32 1]) && all(isfinite(config.input_mean)), 'Invalid input mean.');
    require(isa(config.input_std, 'double') && isequal(size(config.input_std), [32 1]) && all(isfinite(config.input_std)) && all(config.input_std > 0), 'Invalid effective input std.');
    require(isequal(config.command_lower_bounds, [-1;-1]) && isequal(config.command_upper_bounds, [1;1]), 'Command bounds mismatch.');
    require(config.controller_interval_s == 0.1 && metadata.controller_interval_s == 0.1, 'Control interval mismatch.');
    require(isequal(config.reference_preview_s(:), [0.1;0.2]) && isequal(metadata.reference_preview_s(:), [0.1;0.2]), 'Reference preview mismatch.');
    require(config.seed == 42 && config.checkpoint_epoch == 178, 'Wrong frozen baseline checkpoint.');
    require(strcmp(config.checkpoint_sha256, metadata.checkpoint_sha256) && strcmp(config.scaler_sha256, metadata.scaler_sha256) && strcmp(config.schema_sha256, metadata.schema_sha256), 'Source provenance mismatch.');
    require(strcmp(config.schema_sha256, metadata.packaged_schema_sha256), 'Exported schema differs from authoritative source.');
    require(strcmp(config.gate_order, 'reset,update,new') && strcmp(config.normalized_dtype, 'single'), 'GRU gate/dtype mismatch.');
    schema = jsondecode(fileread(fullfile(packageDirectory, 'feature_schema.json')));
    require(numel(schema.supervised_inputs) == 32 && schema.supervised_input_dimension == 32, 'Schema must define 32 inputs.');
    require(isequal([schema.supervised_inputs.index], 0:31), 'Schema indices are not canonical.');
    featureNames = {schema.supervised_inputs.name};
    require(isequal(config.feature_order(:), featureNames(:)), 'Feature order mismatch.');
    stateNames = cellfun(@(name) strtok(name, '['), featureNames(1:22), 'UniformOutput', false);
    require(isequal(config.state_order(:), stateNames(:)), 'State order mismatch.');
    referenceNames = cellfun(@(name) strtok(name, '['), schema.cartesian_order, 'UniformOutput', false);
    require(isequal(config.reference_order(:), referenceNames(:)), 'Reference order mismatch.');
    ok = true;
    if verbose
        fprintf('CONTROLLER PACKAGE: PASS (6898 parameters; 32 inputs; 32 hidden; 2 outputs; 100-ms interval; hashes verified)\n');
    end
catch exception
    fprintf(2, 'CONTROLLER PACKAGE: FAIL - %s\n', exception.message);
    rethrow(exception);
end
end

function require(condition, message)
if ~condition
    error('CraneController:Package', '%s', message);
end
end
