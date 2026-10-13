function command = controller_step(state22, previousCommand, reference01, reference02)
obj = controller_instance();
command = obj.step(state22, previousCommand, reference01, reference02);
end
