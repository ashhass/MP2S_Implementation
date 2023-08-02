$("form[name=login_form]").submit(function(e){
    var $form = $(this);
    var $error = $form.find(".error");
    var data = $form.serialize();

    $.ajax({
        url : "/user/login/",
        type : "POST",
        data : data,
        dataType : "json",
        success : function(resp){
            if (resp[1] == 'monitor') window.location.href = "/dashboard/";
            else window.location.href = "/officer_page/";
            console.log(resp)
        },
        error : function(resp){
            console.log(resp)
            $error.text(resp.responseJSON.error).removeClass("error--hidden");
        }
    });
    e.preventDefault();

}); 

$("form[name=logout_form").submit(function(e){
    var $form = $(this);
    var $error = $form.find(".error");

    $.ajax({
        url : "/user/logout/",
        type : "POST",
        success : function(resp){
            window.location.href = "/";
            console.log(resp)
        },
        error : function(resp){
            console.log(resp)
            $error.text(resp.responseJSON.error.removeClass("error--hidden"));
        }
    });
    e.preventDefault();

}); 


$("form[name=mission").submit(function(e){
    var $form = $(this);
    var data = $form.serialize();
    $.ajax({
        type: 'POST',
        url: '/incident/',
        data: JSON.stringify(data),
        contentType: 'application/json',
        dataType: 'json', 
        success : function(resp){ 
            console.log('SUCCESS'); 
            window.location.href('/officer_page/'); 
        }, 
        error : function(resp){
            console.log('FAILED')
        }
    });

}); 
