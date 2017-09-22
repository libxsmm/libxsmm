(function () {
  'use strict';

  $(document).ready(function () {
    fixSearch();
  });

  /**
   * Please see: https://github.com/rtfd/readthedocs.org/issues/1088
   * This implementation is based on:
   * - https://github.com/gluster/glusterdocs/blob/master/js/fix-docs.js
   * - https://github.com/commercialhaskell/stack/blob/master/doc/js/searchhack.js
   */
  function fixSearch() {
    var target = document.getElementById('rtd-search-form');
    var config = {attributes: true, childList: true};

    var observer = new MutationObserver(function(mutations) {
      observer.disconnect();
      var form = $('#rtd-search-form');
      form.empty();
      // this only serves a single documentation language/version
      form.attr('action', 'https://' + window.location.hostname + '/search.html');
      $('<input>').attr({
        type: "text",
        name: "q",
        placeholder: "Search docs"
      }).appendTo(form);
    });

    if (window.location.origin.indexOf('readthedocs') > -1) {
      observer.observe(target, config);
    }
  }

}());

