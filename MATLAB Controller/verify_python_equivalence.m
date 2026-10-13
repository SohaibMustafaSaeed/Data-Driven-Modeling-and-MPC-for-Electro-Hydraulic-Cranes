function report = verify_python_equivalence(packageDirectory)
if nargin < 1
    packageDirectory = fileparts(mfilename('fullpath'));
end
validate_controller_package(packageDirectory);
input = load(fullfile(packageDirectory, 'verification', 'python_reference_inputs.mat'));
reference = load(fullfile(packageDirectory, 'verification', 'python_reference_outputs.mat'));
summaryFile = fullfile(packageDirectory, 'verification', 'verification_summary.json');
summary = jsondecode(fileread(summaryFile));
controller = CraneNeuralController(packageDirectory);
initial = input.previous_command(1, :);
actual = runSequence(controller, input, initial);
again = runSequence(controller, input, initial);
difference = actual - reference.expected_commands;
report = struct('max_abs_output_difference', max(abs(difference(:))), ...
    'mean_abs_output_difference', mean(abs(difference(:))), ...
    'rmse_output_difference', sqrt(mean(difference(:).^2)), ...
    'per_output_max_error', max(abs(difference), [], 1), ...
    'reset_repeatability_error', max(abs(actual(:) - again(:))), ...
    'sequence_length', size(actual, 1), 'tolerance', summary.tolerance);
report.equivalence_pass = report.max_abs_output_difference <= report.tolerance;
report.reset_repeatability_pass = report.reset_repeatability_error == 0;
summary.matlab_executed = true;
summary.matlab_version = version;
fields = fieldnames(report);
for j = 1:numel(fields)
    summary.(fields{j}) = report.(fields{j});
end
summary.input_validation_pass = testInputs(controller, input);
summary.required_toolboxes = 'Core MATLAB only';
fid = fopen(summaryFile, 'w');
if fid < 0
    error('CraneController:Summary', 'Cannot write verification summary.');
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(summary));
fprintf('PYTHON/MATLAB: max %.9g V; mean %.9g V; RMSE %.9g V; output maxima [%.9g %.9g] V\n', ...
    report.max_abs_output_difference, report.mean_abs_output_difference, report.rmse_output_difference, report.per_output_max_error(1), report.per_output_max_error(2));
fprintf('RESET repeatability: %.9g V; sequence: %d calls; tolerance: %.9g V\n', ...
    report.reset_repeatability_error, report.sequence_length, report.tolerance);
assert(report.equivalence_pass, 'CraneController:Equivalence', 'MATLAB output exceeds the stated Python-equivalence tolerance.');
assert(report.reset_repeatability_pass, 'CraneController:Reset', 'Reset repeatability failed.');
fprintf('PYTHON/MATLAB EQUIVALENCE AND RESET: PASS\n');
end

function result = runSequence(controller, input, initial)
controller.reset(initial);
count = size(input.state22, 1);
result = zeros(count, 2);
for k = 1:count
    result(k, :) = controller.step(input.state22(k, :), input.previous_command(k, :), input.reference_01(k, :), input.reference_02(k, :));
end
end

function ok = testInputs(controller, input)
state = input.state22(1, :); previous = input.previous_command(1, :);
ref1 = input.reference_01(1, :); ref2 = input.reference_02(1, :);
cases = {{state(1:21),previous,ref1,ref2}, {state,previous(1),ref1,ref2}, ...
    {state,previous,ref1(1:3),ref2}, {state,previous,ref1,ref2(1:3)}, ...
    {state + 1i,previous,ref1,ref2}, {reshape(state,2,11),previous,ref1,ref2}};
for slot = 1:4
    for value = [NaN, Inf, -Inf]
        item = {state, previous, ref1, ref2};
        item{slot}(1) = value;
        cases{end + 1} = item;
    end
end
for j = 1:numel(cases)
    item = cases{j};
    failed = false;
    try
        controller.step(item{:});
    catch exception
        failed = strcmp(exception.identifier, 'CraneController:Input');
    end
    assert(failed, 'CraneController:Validation', 'Invalid-input case %d was not rejected correctly.', j);
end
controller.reset();
failed = false;
try
    controller.step(state, [], ref1, ref2);
catch exception
    failed = strcmp(exception.identifier, 'CraneController:InitialPreviousCommand');
end
assert(failed, 'CraneController:Validation', 'Missing initial previous command must fail.');
controller.reset(previous);
row = controller.step(state, [], ref1, ref2);
controller.reset(previous);
column = controller.step(state.', [], ref1.', ref2.');
assert(isequal(row, column), 'CraneController:Validation', 'Row/column interfaces differ.');
ok = true;
fprintf('INPUT VALIDATION: PASS (%d invalid cases; startup and row/column checks)\n', numel(cases));
end
