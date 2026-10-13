classdef CraneNeuralController < handle
    properties (Access = private)
        Weights
        Config
        Hidden
        PreviousCommand
        HasPreviousCommand = false
    end
    methods
        function obj = CraneNeuralController(packageDirectory)
            if nargin < 1
                packageDirectory = fileparts(mfilename('fullpath'));
            end
            [~, obj.Weights, obj.Config] = validate_controller_package(packageDirectory, false);
            obj.reset();
        end

        function reset(obj, previousCommand)
            if nargin < 2 || isempty(previousCommand)
                previous = [];
            else
                previous = obj.vector(previousCommand, 2, 'previousCommand');
            end
            obj.Hidden = zeros(32, 1, 'single');
            obj.PreviousCommand = previous;
            obj.HasPreviousCommand = ~isempty(previous);
        end

        function command = step(obj, state22, previousCommand, reference01, reference02)
            if nargin ~= 5
                error('CraneController:Arguments', 'Supply state22, previousCommand (or []), reference01 and reference02.');
            end
            state = obj.vector(state22, 22, 'state22');
            ref1 = obj.vector(reference01, 4, 'reference01');
            ref2 = obj.vector(reference02, 4, 'reference02');
            if isempty(previousCommand)
                if ~obj.HasPreviousCommand
                    error('CraneController:InitialPreviousCommand', 'First step requires the actual previous applied command in step() or reset().');
                end
                previous = obj.PreviousCommand;
            else
                previous = obj.vector(previousCommand, 2, 'previousCommand');
            end
            physical = [state; previous; ref1; ref2];
            x = single((physical - obj.Config.input_mean) ./ obj.Config.input_std);
            if any(~isfinite(x))
                error('CraneController:Normalization', 'Input normalization exceeds finite single precision. Check input values and units.');
            end
            w = obj.Weights;
            gi = w.gru_weight_ih * x + w.gru_bias_ih;
            gh = w.gru_weight_hh * obj.Hidden + w.gru_bias_hh;
            r = single(1) ./ (single(1) + exp(-(gi(1:32) + gh(1:32))));
            z = single(1) ./ (single(1) + exp(-(gi(33:64) + gh(33:64))));
            n = tanh(gi(65:96) + r .* gh(65:96));
            nextHidden = (obj.Hidden - n) .* z + n;
            head = max(w.head_weight_1 * nextHidden + w.head_bias_1, single(0));
            output = double(tanh(w.head_weight_out * head + w.head_bias_out));
            bounded = min(max(output, obj.Config.command_lower_bounds), obj.Config.command_upper_bounds);
            if any(~isfinite(nextHidden)) || any(~isfinite(bounded))
                error('CraneController:NonfiniteOutput', 'Nonfinite controller result; recurrent state was not advanced.');
            end
            obj.Hidden = nextHidden;
            obj.PreviousCommand = bounded;
            obj.HasPreviousCommand = true;
            command = bounded.';
        end
    end
    methods (Static, Access = private)
        function value = vector(value, count, name)
            if ~isnumeric(value) || ~isreal(value) || ~ismatrix(value) || ~isvector(value) || numel(value) ~= count || any(~isfinite(value(:)))
                error('CraneController:Input', '%s must be a finite real row/column vector with %d values.', name, count);
            end
            value = double(value(:));
        end
    end
end
