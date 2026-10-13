function obj = controller_instance()
persistent instance
if isempty(instance) || ~isvalid(instance)
    instance = CraneNeuralController();
end
obj = instance;
end
