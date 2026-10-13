function reset_controller(previousCommand)
obj = controller_instance();
if nargin < 1
    obj.reset();
else
    obj.reset(previousCommand);
end
end
