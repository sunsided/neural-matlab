function [ a, results ] = feedforward_for_training( network, input )
%FEEDFORWARD Runs the input through the given network.
    
    % prepare the layers
    results = cell(numel(network), 1);
    
    % we begin with the virtual "input layer" whose output is the
    % actual training sample
    current_layer = struct( ...
        'output',   input);
    
    % Loop through all layers in the network
    for j=1:numel(network)
        
        % previously active layer is the previous layer
        previous_layer = current_layer;
        clear current_layer;

        % current input is previous layer's output,
        % augmenting the vector with 1 for the bias unit
        x = [1; previous_layer.output];
        
        % the actual layer function
        z = network{j}.theta * x;
        a = network{j}.sigma(z);
        
        % collect all information required for backpropagation
        current_layer = struct( ...
            'layer',      network{j}, ...
            'input',      x, ...
            'net',        z, ...
            'output',     a);

        % collect all individual layer results for backpropagation
        results{j} = current_layer;
        clear x z a previous_layer;
    end
    
    % The network output is the output of the last layer.
    % Since we leave the above loop on the last layer, the currently 
    % active layer is the last one.
    a = current_layer.output;

end

