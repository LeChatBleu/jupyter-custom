// leave at least 2 line with only a star on it below, or doc generation fails
/**
 *
 *
 * Placeholder for custom user javascript
 * mainly to be overridden in profile/static/custom/custom.js
 * This will always be an empty file in IPython
 *
 * User could add any javascript in the `profile/static/custom/custom.js` file.
 * It will be executed by the ipython notebook at load time.
 *
 * Same thing with `profile/static/custom/custom.css` to inject custom css into the notebook.
 *
 *
 * The object available at load time depend on the version of IPython in use.
 * there is no guaranties of API stability.
 *
 * The example below explain the principle, and might not be valid.
 *
 * Instances are created after the loading of this file and might need to be accessed using events:
 *     define([
 *        'base/js/namespace',
 *        'base/js/events'
 *     ], function(IPython, events) {
 *         events.on("app_initialized.NotebookApp", function () {
 *             IPython.keyboard_manager....
 *         });
 *     });
 *
 * __Example 1:__
 *
 * Create a custom button in toolbar that execute `%qtconsole` in kernel
 * and hence open a qtconsole attached to the same kernel as the current notebook
 *
 *    define([
 *        'base/js/namespace',
 *        'base/js/events'
 *    ], function(IPython, events) {
 *        events.on('app_initialized.NotebookApp', function(){
 *            IPython.toolbar.add_buttons_group([
 *                {
 *                    'label'   : 'run qtconsole',
 *                    'icon'    : 'icon-terminal', // select your icon from http://fortawesome.github.io/Font-Awesome/icons
 *                    'callback': function () {
 *                        IPython.notebook.kernel.execute('%qtconsole')
 *                    }
 *                }
 *                // add more button here if needed.
 *                ]);
 *        });
 *    });
 *
 * __Example 2:__
 *
 * At the completion of the dashboard loading, load an unofficial javascript extension
 * that is installed in profile/static/custom/
 *
 *    define([
 *        'base/js/events'
 *    ], function(events) {
 *        events.on('app_initialized.DashboardApp', function(){
 *            require(['custom/unofficial_extension.js'])
 *        });
 *    });
 *
 * __Example 3:__
 *
 *  Use `jQuery.getScript(url [, success(script, textStatus, jqXHR)] );`
 *  to load custom script into the notebook.
 *
 *    // to load the metadata ui extension example.
 *    $.getScript('/static/notebook/js/celltoolbarpresets/example.js');
 *    // or
 *    // to load the metadata ui extension to control slideshow mode / reveal js for nbconvert
 *    $.getScript('/static/notebook/js/celltoolbarpresets/slideshow.js');
 *
 *
 * @module IPython
 * @namespace IPython
 * @class customjs
 * @static
 */

// stackoverflow: Disable Ctrl+Enter sublime keymap in jupyter notebook
 require(["codemirror/keymap/sublime", "notebook/js/cell", "base/js/namespace"],
 function(sublime_keymap, cell, IPython) {
     cell.Cell.options_default.cm_config.keyMap = 'sublime';
     cell.Cell.options_default.cm_config.extraKeys["Ctrl-Enter"] = function(cm) {}
     var cells = IPython.notebook.get_cells();
     for(var cl=0; cl< cells.length ; cl++){
         cells[cl].code_mirror.setOption('keyMap', 'sublime');
         cells[cl].code_mirror.setOption("extraKeys", {
             "Ctrl-Enter": function(cm) {}
         });
     }
 } 
);

// Register a global action (navigation(menu) bar) 토글 기능 추가
var action_name = Jupyter.actions.register({
    help: 'hide/show the menubar',
    handler : function(env) {
        $('#menubar').toggle();
        events.trigger('resize-header.Page');
    }
}, 'toggle-menubar', 'jupyter-notebook');
// Add a menu item to the View menu
$('#view_menu').prepend('<li id="toggle_menu" title="Show/Hide the menu bar"><a href="#">Toggle Menu</a></li>').click(function() {
    Jupyter.actions.call(action_name);
});
// Add a shortcut: CMD+M (or CTRL+M on Windows) to toggle menu bar
Jupyter.keyboard_manager.command_shortcuts.add_shortcut('N', action_name);

// nbextensions Snippets Menu 커스텀
require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
  console.log("Loading `snippets_menu` customizations from `custom.js`");
  var horizontal_line = "---";
  var scikit_learn = {
    name: "scikit_learn",
    "sub-menu": [
      {
        name: "SGDClassifier",
        snippet: [
          "from sklearn.linear_model import SGDClassifier",
          "",
          "sgd_clf = SGDClassifier()",
        ],
      },
      {
        name: "model_selection",
        "sub-menu": [
          {
            name: "cross_val_predict",
            snippet: [
              "from sklearn.model_selection import cross_val_predict",
              "",
              "y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)  # 예측값을 반환",
              "y_train_scores = cross_val_predict(model, X_train, y_train, cv=3,",
              "                                method = 'decision_function')  # 점수를 반환",
              "                                                               # 여기서 점수는 분류기의 성능이 아니라, 분류에 사용할 점수",
            ],
          },
          {
            name: "cross_val_score",
            snippet: [
              "from sklearn.model_selection import cross_val_score",
              "",
              "cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')  # model의 성능 점수를 반환",
            ],
          },
        ],
      },
      {
        name: "preprocessing",
        "sub-menu": [
          {
            name: "StandardScaler",
            snippet: [
              "from sklearn.preprocessing import StandardScaler",
              "",
              "scaler = StandardScaler()",
              "X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))",
            ],
          },
        ],
      },
      {
        name: "svm",
        "sub-menu": [
          {
            name: "SVC",
            snippet: [
              "from sklearn.svm import SVC",
              "",
              "svm_clf = SVC(gamma='auto', random_state=42)",
              "svm_clf.fit(X_train, y_train)",
              "svm_clf.predict(X_train)",
            ],
          },
        ],
      },
      {
        name: "neighbors",
        "sub-menu": [
          {
            name: "KNeighborsClassifier",
            snippet: [
              "from sklearn.neighbors import KNeighborsClassifier",
              "",
              "knn_clf = KNeighborsClassifier()",
              "knn_clf.fit(X_train, y_train)",
              "knn_clf.predict(X_train)",
            ],
          },
        ],
      },
      {
        name: "multiclass",
        "sub-menu": [
          {
            name: "OneVsRestClassifier",
            snippet: [
              "# OvO대신 OvR 전략을 사용하는 분류기",
              "from sklearn.multiclass import OneVsRestClassifier",
              "",
              "ovr_clf = OneVsRestClassifier(SVC(gamma = 'auto', random_state = 42))  # 다중 분류에서 SVC가 OvO가 아닌 OvR 전략을 사용하도록 강제",
              "ovr_clf.fit(X_train, y_train)",
              "ovr_clf.predict(X_train)",
            ],
          },
        ],
      },
      {
        name: "ensemble",
        "sub-menu": [
          {
            name: "RandomForestClassifier",
            snippet: [
              "from sklearn.ensemble import RandomForestClassifier",
              "",
              "forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)",
              "y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,",
              "                                    method='predict_proba')",
            ],
          },
        ],
      },
      {
        name: "metrics",
        "sub-menu": [
          {
            name: "confusion_matrix",
            snippet: [
              "from sklearn.metircs import confusion_matrix",
              "",
              "confusion_matrix(y_train, y_train_pred)",
            ],
          },
          {
            name: "precision_score",
            snippet: [
              "from sklearn.metrics import precision_score",
              "",
              "precision_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "recall_score",
            snippet: [
              "from sklearn.metrics import recall_score",
              "",
              "recall_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "f1_score",
            snippet: [
              "from sklearn.metrics import f1_score",
              "",
              "f1_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "precision_recall_curve",
            snippet: [
              "from sklearn.metrics import precision_recall_curve",
              "",
              "precision, recalls, thresholds = precision_recall_curve(y_bp, y_bp_scores)",
            ],
          },
          {
            name: "roc_curve",
            snippet: [
              "from sklearn.metrics import roc_curve",
              "",
              "fpr, tpr, thresholds = roc_curve(y_bp, y_bp_scores)  # y_bp_scores는 classifier의 decision_function()으로 구함",
              "                                                     # decision_function()이 없는 classifier의 경우,",
              "                                                     # predict_proba()로 확률을 구하고 양성 클래스일 확률을 점수 대신 사용",
              "",
              "def plot_roc_curve(fpr, tpr, label=None):",
              "    plt.plot(fpr, tpr, linewidth=2, label=label)",
              "    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선",
              "    plt.axis([0, 1, 0, 1])",
              "    plt.xlabel('False Positive Rate (Fall-Out)', fontsize = 16)",
              "    plt.ylabel('True Positive Rate (Recall)', fontsize = 16)",
              "    plt.grid(True)",
              "",
              "plt.figure(figsize=(8, 6))",
              "plot_roc_curve(fpr, tpr)",
              "fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]",
              "plt.plot([fpr_90, fpr_90], [0., recall_90_precision], 'r:')",
              "plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], 'r:')",
              "plt.plot([fpr_90], [recall_90_precision], 'ro')",
              "plt.show()",
            ],
          },
          {
            name: "roc_auc_score",
            snippet: [
              "from sklearn.metrics import roc_auc_score",
              "",
              "roc_auc_score(y_bp, y_bp_scores)  # y_bp_scores는 classifier의 decision_function()이나 predict_proba()로 계산",
            ],
          },
        ],
      },
      {
        name: "markdown",
        "sub-menu": [
          {
            name: "md: confusion_matrix",
            snippet: [
              "|실제|negative로 분류|positive로 분류|",
              "|---|---|---|",
              "|negative class| true negative(진짜 음성) | false positive(거짓 양성) |",
              "|positive class| false negative(거짓 음성) | true positive(진짜 양성) |",
            ],
          },
        ],
      },
    ],
  };
  var my_favorites = {
    name: "My $\\nu$ favorites",
    "sub-menu": [
      {
        name: "Multi-line snippet",
        snippet: [
          "new_command(3.14)",
          "",
          'other_new_code_on_new_line("with a string!")',
          "stringy('escape single quotes once')",
          "stringy2('or use single quotes inside of double quotes')",
          'backslashy("This \\ appears as just one backslash in the output")',
          'backslashy2("Here are \\\\ two backslashes")',
        ],
      },
      {
        name: "code test",
        snippet: [
          "import unittest",
          "",
          "class bp_TestCase(unittest.TestCase):",
          "    def test_bp(self):",
          "        self.assertEqual()",
          "        self.assertNotEqual()",
          "        self.assertTrue()",
          "        self.assertFalse()",
          "        self.assertIn()",
          "        self.assertNotIn()",
          "",
          "if __name__ == '__main__':",
          "    unittest.main()",
        ],
      },
    ],
  };
  snippets_menu.options["menus"].push(snippets_menu.default_menus[0]);
  snippets_menu.options["menus"][0]["sub-menu"].push(horizontal_line);
  snippets_menu.options["menus"][0]["sub-menu"].push(scikit_learn);
  snippets_menu.options["menus"][0]["sub-menu"].push(my_favorites);
  console.log("Loaded `snippets_menu` customizations from `custom.js`");
});