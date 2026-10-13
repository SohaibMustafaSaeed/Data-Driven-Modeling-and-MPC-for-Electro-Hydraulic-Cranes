function command = example_controller_run()
folder = fileparts(mfilename('fullpath'));
validate_controller_package(folder);
fixture = load(fullfile(folder, 'verification', 'python_reference_inputs.mat'));
ctrl = CraneNeuralController(folder);
previousCommand = fixture.previous_command(1, :);
ctrl.reset(previousCommand);
command = ctrl.step(fixture.state22(1, :), previousCommand, fixture.reference_01(1, :), fixture.reference_02(1, :));
fprintf('Fixture API example only; u_lift=%.9g V, u_jib=%.9g V\n', command);
end
