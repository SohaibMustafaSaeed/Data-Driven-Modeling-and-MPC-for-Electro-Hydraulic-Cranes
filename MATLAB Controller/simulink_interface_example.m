function simulink_interface_example(readState, readPreview, applyAndAdvance, initialPreviousCommand, numberOfSteps)
% Callback template only: not connected to a Simulink model.
% readState() returns 22 current physical signals in canonical order.
% readPreview(t) returns desired x/z/vx/vz at absolute simulation time t.
% applyAndAdvance(command, interval) applies/holds two commands and returns
% the actual applied command after advancing the external plant interval.
if nargin ~= 5
    error('CraneController:Example', 'Supply external readState/readPreview/applyAndAdvance callbacks, actual initial command and number of control steps.');
end
initialize_controller();
reset_controller(initialPreviousCommand);
previousCommand = initialPreviousCommand;
for k = 1:numberOfSteps
    t = (k - 1) * 0.1;
    state22 = readState();
    reference01 = readPreview(t + 0.1);
    reference02 = readPreview(t + 0.2);
    command = controller_step(state22, previousCommand, reference01, reference02);
    % The external plant uses its own finer integration timestep.
    previousCommand = applyAndAdvance(command, 0.1);
end
end
