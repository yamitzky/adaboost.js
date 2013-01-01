function valueOf(id) {
  return document.getElementById(id).value;
}
function random_range(min, max) {
  return Math.random() * (max - min) + min;
}
function boundary(x) {
  return eval(valueOf('function'));
}
function classify(x) {
  if (x[1] > boundary(x[0])) {
    return +1;
  } else {
    return -1;
  }
}

function learn() {
  var x_min = parseFloat(valueOf('x_min'), 10);
  var x_max = parseFloat(valueOf('x_max'), 10);
  var y_min = parseFloat(valueOf('y_min'), 10);
  var y_max = parseFloat(valueOf('y_max'), 10);

  // training data
  x_training = [];
  y_training = [];
  for (var i = 0; i < 1000; i++) {
    x_training[i] = [
      random_range(x_min, x_max),
      random_range(y_min, y_max),
      ];
    y_training[i] = classify(x_training[i]);
  }

  // learn
  ab = new AdaBoost();
  ab.learn(x_training, y_training, parseInt(valueOf('ab_length'), 10));
  test();
}

function test() {
  var x_min = parseFloat(valueOf('x_min'), 10);
  var x_max = parseFloat(valueOf('x_max'), 10);
  var y_min = parseFloat(valueOf('y_min'), 10);
  var y_max = parseFloat(valueOf('y_max'), 10);
  var h_rate = parseFloat(valueOf('h_rate'), 10);
  var error = 0;

  var data = new google.visualization.DataTable();
  data.addColumn('number', 'x0');
  data.addColumn('number', '+1');
  data.addColumn('number', '-1');
  data.addColumn('number', 'line');
  // test
  for (var i = 0, len = 1000; i < len; i++) {
    var x = [
      random_range(x_min, x_max),
      random_range(y_min, y_max),
      ];
    if (ab) {
      var y = ab.classify(x, ab.D.length * h_rate);
      if (y > 0) {
        data.addRow([x[0], x[1], null, null]);
      } else {
        data.addRow([x[0], null, x[1], null]);
      }
      if (y != classify(x)) {
        error++;
      }
    } else {
      data.addRow([x[0], x[1], null, null]);
    }
  }
  // draw training line
  for (var i = 0, len = 100; i < len; i++) {
    var x = (x_max - x_min) / len * i + x_min;
    data.addRow([x, null, null, boundary(x)]);
  }

  chart.draw(data, {
    title: 'AdaBoost', width: 320, height: 320,
    tooltip: {trigger: 'none'},
    pointSize: 2,
    series: [{}, {}, {pointSize:0, lineWidth:2, color:"black"}]
  });

  document.getElementById('error').innerText = 'error rate: ' + (error / 1000).toString();
}

function main() {
  var methods = ["E","LN2","LN10","LOG2E","LOG10E","PI","SQRT1_2","SQRT2","abs","acos","asin","atan","atan2","ceil","cos","cosh","exp","floor","log","max","min","pow","random","round","sin","sinh","sqrt","tan","tanh"];
  for (var i = 0, len = methods.length; i < len; i++) {
    var method = methods[i];
    window[method] = Math[method];
  }

  chart = new google.visualization.ScatterChart(
      document.getElementById('canvas'));
  test();
}

ab = undefined;
google.load('visualization', '1.0', {'packages':['corechart']});
google.setOnLoadCallback(main);
