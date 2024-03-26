class ConfigValidation:
    def __init__(self, config):
        self.config = config

    # Validate
    def validate_convolutions(self, l_in=60, max=2000):
        ae_conv_num = self.config.get('ae_conv_num', 0)
        ae_conv_kernel = self.config.get('ae_conv_kernel', 3)
        ae_conv_stride = self.config.get('ae_conv_stride', 1)
        ae_conv_padding = self.config.get('ae_conv_padding', 0)

        def l_out(l_in, kernel, stride, padding, dilation=1):
            return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        input_size = l_in
        for i in range(ae_conv_num):
            input_size = l_out(input_size, ae_conv_kernel, ae_conv_stride, ae_conv_padding)
        
        if input_size > max:
            return True
        return False
    
    def validate_multichoice(self, key, value):
        property_content = self.config[key] if key in self.config else []
        if value['tune_function'] == 'multichoice':
            property_content = []
            for parameter in value['tune_parameters']:
                multichoice_key = f"MC-{key}-{parameter}"
                if self.config[multichoice_key] == 1:
                    property_content.append(parameter)
            if property_content == []:
                return {
                    'score': float('-inf'),
                    'num_params': -1,
                    'num_trainable_params': -1,
                    'error_type': '0 reducer',
                    'error_message': 'There is no reducer in the configuration.',
                    'error_traceback': 'There is no reducer in the configuration.'
                }
        return property_content
    
    # def validate_convolution_60_2000(self):
    #     return self.validate_convolutions(l_in=60, max=2000)
    
    # Call dynamic function
    def execute_validation(self, tag: str, params):
        function_name = f'validate_{tag}'
        if hasattr(self, function_name) and callable(function_to_call := getattr(self, function_name)):
            function_to_call(**params)