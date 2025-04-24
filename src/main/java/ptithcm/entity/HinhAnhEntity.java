package ptithcm.entity;

import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Table;

@Entity
@Table(name="HINHANH")
public class HinhAnhEntity{
	@Id
	@GeneratedValue
	@Column(name = "MAHINHANH")
	private Integer maHinhAnh;
	@Column(name = "LINK")
	private String link;
//	@ManyToOne
//	@JoinColumn(name="MASP")
//	private SanPhamEntity sanPham;
	@OneToMany(mappedBy = "hinhAnh", fetch = FetchType.LAZY)
	private List<SanPhamEntity> sanPhams;
	public Integer getMaHinhAnh() {
		return maHinhAnh;
	}
	public String getLink() {
		return link;
	}
	public void setLink(String link) {
		this.link = link;
	}
	public List<SanPhamEntity> getSanPhams() {
		return sanPhams;
	}
	public void setSanPhams(List<SanPhamEntity> sanPhams) {
		this.sanPhams = sanPhams;
	}
	public void setMaHinhAnh(Integer maHinhAnh) {
		this.maHinhAnh = maHinhAnh;
	}


	
}