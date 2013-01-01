var AdaBoost, LinearDevice;

Math.sinh = function (x) {
  return (Math.exp(x) - Math.exp(-x)) / 2;
};

Math.cosh = function(x) {
  return (Math.exp(x) + Math.exp(-x)) / 2;
};

Math.tanh = function(x) {
  var p = Math.exp(x);
  var n = Math.exp(-x);
  return (p - n) / (p + n);
};

LinearDevice = (function() {
  function LinearDevice(dimension) {
    this.weight = [];

    var d = dimension + 1;
    for (var i = 0; i < d; i++) {
      this.weight[i] = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
  }

  LinearDevice.prototype.calc = function(x) {
    var wx = 0.0; // multiply w_vector x_vector
    var d = this.weight.length;
    for (var k = 0; k < d; k++) {
      var x_bias = k == 0 ? 1 : x[k-1];
      wx += this.weight[k] * x_bias;
    }
    return Math.tanh(wx);
  };

  LinearDevice.prototype.learnWithSteepestDescentMethod =
    function(x_training, y_training, D, grad_times, grad_rate) {
      var i, j, k;
      var d = this.weight.length;
      var m = x_training.length;
      for (j = 0; j < grad_times; j++) {
        // update w of linear_device
        for (i = 0; i < m; i++) {
          var tanhwx = this.calc(x_training[i]);
          for (k = 0; k < d; k++) {
            // dE/dw
            var x_bias = k == 0 ? 1 : x_training[i][k-1];
            this.weight[k] += grad_rate * 2 * D[i] *
              (y_training[i] - tanhwx) * (1 - tanhwx * tanhwx) * x_bias;
          }
        }
      }
    };

  return LinearDevice;
})();

AdaBoost = (function() {
  function AdaBoost() {
    this.D = [];
    this.alpha = [];
    this.linear_devices = [];
  }

  AdaBoost.prototype.learn = function(x_training, y_training, ab_times) {
    var i, t;
    var T = ab_times;
    var m = x_training.length;
    var d = x_training[0].length; // dimension of X
    var D_0 = [];

    console.log("data length: " + m);

    // init D_1
    for (i = 0; i < m; i++) {
      D_0[i] = 1.0 / m;
    }
    this.D[0] = D_0;

    for (t = 0; t < T; t++) {
      var epsilon = 0.0;
      var z_t;
      var D_t1 = [];
      var h_t = [];
      var exp_i = [];
      var linear_device = new LinearDevice(d);

      // weak learning
      linear_device.learnWithSteepestDescentMethod(
          x_training, y_training, this.D[t], 100, 0.01);
      this.linear_devices[t] = linear_device;

      //cache h_t
      for (i = 0; i < m; i++) {
        h_t[i] = linear_device.calc(x_training[i]);
      }

      epsilon = 0.0;
      for (i = 0; i < m; i++) {
        epsilon += this.D[t][i] * Math.abs(y_training[i] - h_t[i]);
      }
      epsilon /= 2.0;

      this.alpha[t] = 1.0 / 2.0 * Math.log((1 - epsilon) / epsilon);
      z_t = 0.0;
      for (i = 0; i < m; i++) {
        exp_i[i] = Math.exp(-this.alpha[t] * y_training[i] * h_t[i]);
        z_t += this.D[t][i] * exp_i[i];
      }
      // determine next D_t+1
      for (i = 0; i < m; i++) {
        D_t1[i] = this.D[t][i] * exp_i[i] / z_t;
      }
      this.D[t+1] = D_t1;
    }
    console.log("Learning ended");
  };

  AdaBoost.prototype.classify = function(x, _T) {
    var result = 0.0;
    var T = (_T !== undefined) ? _T : this.linear_devices.length;
    for (var t = 0; t < T; t++) {
      result += this.alpha[t] * this.linear_devices[t].calc(x);
    }
    return (result > 0) ? +1 : -1;
  };

  return AdaBoost;
})();
