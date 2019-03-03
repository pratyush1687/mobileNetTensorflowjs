$(()=>{
    $.get('http://localhost:3000/getData',(data)=>{
        let location = data.location;
        window.location = location;    
    })
})